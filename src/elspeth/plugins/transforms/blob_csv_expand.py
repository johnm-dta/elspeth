"""Expand a payload-store CSV blob into pipeline rows."""

from __future__ import annotations

import codecs
import copy
import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.errors import FrameworkBugError, TransformErrorReason
from elspeth.contracts.payload_store import IntegrityError, PayloadNotFoundError
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.sources.field_normalization import ExternalHeaderError, resolve_field_names

DEFAULT_MAX_OUTPUT_ROWS = 100_000
DEFAULT_MAX_BLOB_BYTES = 100 * 1024 * 1024


class BlobCSVExpandConfig(TransformDataConfig):
    """Configuration for blob_csv_expand."""

    blob_ref_field: str = Field(default="blob_ref", description="Input field containing a payload-store content hash.")
    delimiter: str = Field(default=",", description="Single-character delimiter used to split CSV fields.")
    encoding: str = Field(default="utf-8", description="Encoding used to decode the CSV blob.")
    skip_rows: int = Field(default=0, ge=0, description="Number of leading CSV records to skip before reading headers or data.")
    columns: list[str] | None = Field(default=None, description="Explicit normalized column names for headerless CSV blobs.")
    field_mapping: dict[str, str] | None = Field(
        default=None, description="Optional mapping from observed CSV headers to normalized names."
    )
    include_row_index: bool = Field(default=True, description="Whether to emit the row index within the parsed CSV document.")
    row_index_field: str = Field(default="csv_row_index", description="Output field receiving the zero-based CSV data row index.")
    max_output_rows: int = Field(default=DEFAULT_MAX_OUTPUT_ROWS, gt=0, description="Maximum rows emitted for one input blob.")
    max_blob_bytes: int = Field(default=DEFAULT_MAX_BLOB_BYTES, gt=0, description="Maximum payload size accepted from the blob store.")

    @field_validator("blob_ref_field")
    @classmethod
    def _reject_empty_blob_ref_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blob_ref_field must not be empty")
        return value.strip()

    @field_validator("delimiter")
    @classmethod
    def _validate_delimiter(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError(f"delimiter must be a single character, got {value!r}")
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        try:
            codecs.lookup(value)
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {value!r}") from exc
        return value

    @field_validator("row_index_field")
    @classmethod
    def _validate_row_index_field(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("row_index_field must not be empty")
        if not stripped.isidentifier():
            raise ValueError(f"row_index_field must be a valid Python identifier, got {value!r}")
        return stripped

    @model_validator(mode="after")
    def _validate_normalization_options(self) -> BlobCSVExpandConfig:
        from elspeth.contracts.identifiers import validate_field_names

        if self.columns is not None:
            validate_field_names(self.columns, "columns")
        if self.field_mapping is not None and self.field_mapping:
            validate_field_names(list(self.field_mapping.values()), "field_mapping values")
        if self.include_row_index and self.row_index_field == self.blob_ref_field:
            raise ValueError(f"row_index_field {self.row_index_field!r} collides with blob_ref_field")
        return self

    @property
    def declared_input_fields(self) -> frozenset[str]:
        return super().declared_input_fields | frozenset({self.blob_ref_field})


@dataclass(frozen=True, slots=True)
class _ParsedCSV:
    rows: tuple[Mapping[str, object], ...]
    headers: tuple[str, ...]


class _BlobCSVParseError(Exception):
    def __init__(self, reason: TransformErrorReason) -> None:
        super().__init__(str(reason))
        self.reason = reason


def _csv_error_reason(reason: str, **details: object) -> TransformErrorReason:
    return cast(TransformErrorReason, {"reason": reason, **details})


def _blob_csv_added_output_fields(cfg: BlobCSVExpandConfig) -> tuple[FieldDefinition, ...]:
    fields: list[FieldDefinition] = []
    if cfg.columns is not None:
        fields.extend(FieldDefinition(name=column, field_type="str", required=True) for column in cfg.columns)
    if cfg.include_row_index:
        fields.append(FieldDefinition(name=cfg.row_index_field, field_type="int", required=True))
    return tuple(fields)


def _build_blob_csv_output_schema_config(schema_config: SchemaConfig, cfg: BlobCSVExpandConfig) -> SchemaConfig:
    field_by_name: dict[str, FieldDefinition] = {}
    if schema_config.fields is not None:
        field_by_name.update((field.name, field) for field in schema_config.fields)

    added_fields = _blob_csv_added_output_fields(cfg)
    field_by_name.update((field.name, field) for field in added_fields)

    base_guaranteed = set(schema_config.guaranteed_fields or ())
    output_guaranteed = base_guaranteed | {field.name for field in added_fields}

    return SchemaConfig(
        mode=schema_config.mode if schema_config.fields is not None else "flexible",
        fields=tuple(field_by_name.values()),
        guaranteed_fields=tuple(sorted(output_guaranteed)) if output_guaranteed else schema_config.guaranteed_fields,
        audit_fields=schema_config.audit_fields,
        required_fields=schema_config.required_fields,
    )


class BlobCSVExpand(BaseTransform):
    """Parse a CSV blob and emit one output row per CSV data row."""

    name = "blob_csv_expand"
    determinism = Determinism.IO_READ
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:83d598380cb107aa"
    config_model = BlobCSVExpandConfig
    creates_tokens = True
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "blob_ref_field": "blob_ref",
        }

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        cfg = BlobCSVExpandConfig.from_dict(options, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)

        self._blob_ref_field = cfg.blob_ref_field
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding
        self._skip_rows = cfg.skip_rows
        self._columns = tuple(cfg.columns) if cfg.columns is not None else None
        self._field_mapping = cfg.field_mapping
        self._include_row_index = cfg.include_row_index
        self._row_index_field = cfg.row_index_field
        self._max_output_rows = cfg.max_output_rows
        self._max_blob_bytes = cfg.max_blob_bytes

        self.declared_output_fields = frozenset(field.name for field in _blob_csv_added_output_fields(cfg))

        self.input_schema = create_schema_from_config(cfg.schema_config, "BlobCSVExpandInput", allow_coercion=False)
        self._output_schema_config = _build_blob_csv_output_schema_config(cfg.schema_config, cfg)
        self.output_schema = create_schema_from_config(self._output_schema_config, "BlobCSVExpandOutput", allow_coercion=False)

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Expand a payload-store CSV blob into one output row per CSV data row while preserving upstream fields.",
                composer_hints=(
                    "Use blob_csv_expand after blob_fetch or another transform that emits a payload-store blob_ref.",
                    "The upstream URL or document id field is preserved on every emitted row for disambiguation.",
                    "CSV headers are normalized like the csv source; use columns for headerless blobs and field_mapping for overrides.",
                    "If a CSV column collides with an existing input field such as url, rename upstream or use field_mapping before expanding.",
                ),
            )
        return None

    def on_start(self, ctx: LifecycleContext) -> None:
        super().on_start(ctx)
        if ctx.payload_store is None:
            raise FrameworkBugError("BlobCSVExpand requires payload_store — orchestrator must configure it before on_start().")
        self._payload_store = ctx.payload_store

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        del ctx
        blob_ref = row[self._blob_ref_field]
        if type(blob_ref) is not str:
            raise TypeError(
                f"Field '{self._blob_ref_field}' must be a string payload-store hash, got {type(blob_ref).__name__}. "
                "This indicates an upstream validation bug."
            )

        try:
            body = self._payload_store.retrieve(blob_ref)
        except PayloadNotFoundError as exc:
            return TransformResult.error(
                {
                    "reason": "blob_not_found",
                    "field": self._blob_ref_field,
                    "blob_ref": blob_ref,
                    "error": str(exc),
                },
                retryable=False,
            )
        except IntegrityError:
            raise

        body_size = len(body)
        if body_size > self._max_blob_bytes:
            return TransformResult.error(
                {
                    "reason": "blob_too_large",
                    "field": self._blob_ref_field,
                    "blob_ref": blob_ref,
                    "body_size": body_size,
                    "max_blob_bytes": self._max_blob_bytes,
                },
                retryable=False,
            )

        try:
            text = body.decode(self._encoding)
        except UnicodeDecodeError as exc:
            return TransformResult.error(
                {
                    "reason": "decode_failed",
                    "field": self._blob_ref_field,
                    "blob_ref": blob_ref,
                    "encoding": self._encoding,
                    "error": str(exc),
                },
                retryable=False,
            )

        base = row.to_dict()
        try:
            parsed = self._parse_csv(text, base_fields=frozenset(base))
        except _BlobCSVParseError as exc:
            return TransformResult.error(exc.reason, retryable=False)

        output_rows: list[dict[str, Any]] = []
        for row_index, csv_row in enumerate(parsed.rows):
            output = copy.deepcopy(base)
            output.update(csv_row)
            if self._include_row_index:
                output[self._row_index_field] = row_index
            output_rows.append(output)

        if not output_rows:
            return TransformResult.error(
                {
                    "reason": "empty_csv",
                    "field": self._blob_ref_field,
                    "blob_ref": blob_ref,
                    "error": "CSV blob had no data rows",
                },
                retryable=False,
            )

        first_keys = set(output_rows[0])
        for index, output_row in enumerate(output_rows[1:], start=1):
            row_keys = set(output_row)
            if row_keys != first_keys:
                raise ValueError(
                    f"Multi-row output has heterogeneous schema: row 0 has fields {sorted(first_keys)}, "
                    f"row {index} has fields {sorted(row_keys)}"
                )

        output_contract = narrow_contract_to_output(input_contract=row.contract, output_row=output_rows[0])
        output_contract = self._apply_declared_output_field_contracts(output_contract)
        output_contract = self._align_output_contract(output_contract)

        return TransformResult.success_multi(
            [PipelineRow(output, output_contract) for output in output_rows],
            success_reason={
                "action": "expanded_blob",
                "fields_added": sorted(set(parsed.headers) | ({self._row_index_field} if self._include_row_index else set())),
                "metadata": {
                    "blob_ref": blob_ref,
                    "row_count": len(output_rows),
                },
            },
        )

    def _parse_csv(self, text: str, *, base_fields: frozenset[str]) -> _ParsedCSV:
        stream = io.StringIO(text, newline="")
        reader = csv.reader(stream, delimiter=self._delimiter, strict=True)

        for skip_idx in range(self._skip_rows):
            try:
                if next(reader, None) is None:
                    raise _BlobCSVParseError(
                        _csv_error_reason(
                            "csv_exhausted_during_skip_rows",
                            skip_rows=self._skip_rows,
                            rows_skipped=skip_idx,
                        )
                    )
            except csv.Error as exc:
                raise _BlobCSVParseError(
                    _csv_error_reason(
                        "csv_parse_error",
                        phase="skip_rows",
                        row_number=skip_idx + 1,
                        line_number=reader.line_num,
                        error=str(exc),
                    )
                ) from exc

        def next_nonblank_record() -> list[str] | None:
            while True:
                values = next(reader, None)
                if values is None:
                    return None
                if values:
                    return values

        if self._columns is not None:
            raw_headers = None
        else:
            try:
                raw_headers = next_nonblank_record()
            except csv.Error as exc:
                raise _BlobCSVParseError(
                    _csv_error_reason(
                        "csv_parse_error",
                        phase="header",
                        line_number=reader.line_num,
                        error=str(exc),
                    )
                ) from exc
            if raw_headers is None:
                return _ParsedCSV(rows=(), headers=())

        try:
            field_resolution = resolve_field_names(
                raw_headers=raw_headers,
                field_mapping=self._field_mapping,
                columns=list(self._columns) if self._columns is not None else None,
            )
        except ExternalHeaderError as exc:
            raise _BlobCSVParseError(_csv_error_reason("csv_header_error", error=str(exc))) from exc
        except ValueError as exc:
            raise _BlobCSVParseError(_csv_error_reason("csv_config_error", error=str(exc))) from exc

        headers = tuple(field_resolution.final_headers)
        collisions = sorted(base_fields & set(headers))
        if self._include_row_index and self._row_index_field in base_fields:
            collisions.append(self._row_index_field)
        if self._include_row_index and self._row_index_field in headers:
            collisions.append(self._row_index_field)
        if collisions:
            raise _BlobCSVParseError(
                _csv_error_reason(
                    "field_collision",
                    fields=sorted(set(collisions)),
                    error="CSV output fields collide with existing input fields",
                )
            )

        expected_count = len(headers)
        rows: list[dict[str, Any]] = []
        row_number = 0
        while True:
            try:
                values = next_nonblank_record()
            except csv.Error as exc:
                raise _BlobCSVParseError(
                    _csv_error_reason(
                        "csv_parse_error",
                        phase="data",
                        line_number=reader.line_num,
                        row_number=row_number + 1,
                        error=str(exc),
                    )
                ) from exc
            if values is None:
                break
            row_number += 1
            if row_number > self._max_output_rows:
                raise _BlobCSVParseError(
                    _csv_error_reason(
                        "too_many_rows",
                        row_count=row_number,
                        max_output_rows=self._max_output_rows,
                    )
                )
            if len(values) != expected_count:
                raise _BlobCSVParseError(
                    _csv_error_reason(
                        "csv_column_count_mismatch",
                        line_number=reader.line_num,
                        row_number=row_number,
                        expected=expected_count,
                        actual=len(values),
                    )
                )
            rows.append(dict(zip(headers, values, strict=False)))

        return _ParsedCSV(rows=tuple(rows), headers=headers)

    def close(self) -> None:
        pass
