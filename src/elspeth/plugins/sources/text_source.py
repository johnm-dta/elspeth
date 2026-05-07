"""Plain-text line source plugin for ELSPETH.

Reads ``.txt`` / ``.md`` style line-oriented files and emits one row per line
into a single configured column.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.
"""

from __future__ import annotations

import codecs
import keyword
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

from pydantic import ValidationError, field_validator

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.contracts.contexts import SourceContext
from elspeth.contracts.contract_builder import ContractBuilder
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.plugins.infrastructure.base import BaseSource
from elspeth.plugins.infrastructure.config_base import SourceDataConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config


def _contains_surrogateescape_chars(value: str) -> bool:
    """Return True when value contains surrogateescape-decoded bytes."""
    return any(0xDC80 <= ord(char) <= 0xDCFF for char in value)


def _surrogateescape_line_to_bytes(value: str, encoding: str) -> bytes:
    """Encode a surrogateescape-decoded line back to bytes for quarantine."""
    try:
        return value.encode(encoding, errors="surrogateescape")
    except UnicodeEncodeError:
        return value.encode("utf-8", errors="surrogateescape")


class TextSourceConfig(SourceDataConfig):
    """Configuration for the plain-text line source plugin."""

    column: str
    encoding: str = "utf-8"
    strip_whitespace: bool = True
    skip_blank_lines: bool = True

    @field_validator("column")
    @classmethod
    def _validate_column(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(f"column '{v}' is not a valid Python identifier")
        if keyword.iskeyword(v):
            raise ValueError(f"column '{v}' is a Python keyword")
        return v

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, v: str) -> str:
        try:
            codecs.lookup(v)
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {v!r}") from exc
        return v


class TextSource(BaseSource):
    """Load one output row per text line into a configured column."""

    name = "text"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:c76a93b112ac51ca"
    config_model = TextSourceConfig
    _on_validation_failure: str

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = TextSourceConfig.from_dict(config, plugin_name=self.name)
        schema_config = cfg.schema_config
        stored_config = config

        # Auto-declare {column} as a guaranteed output field only for the
        # shared observed-text contract case: observed schema, no explicit
        # guaranteed_fields, non-empty column. TextSource always produces
        # {column: value} for every row, so this is a provable invariant.
        # Keep this narrow: fixed/flexible schemas already express their
        # guarantees through normal SchemaConfig semantics and must not be
        # rewritten into an explicit guaranteed_fields declaration.
        if schema_config.mode == "observed" and not schema_config.declares_guaranteed_fields and cfg.column:
            schema_config = replace(
                schema_config,
                guaranteed_fields=(cfg.column,),
            )
            # DAG builder currently reads source.config["schema"], not the
            # plugin's private _schema_config. Only the observed/no-explicit-
            # guarantees path needs a copied raw-config rewrite so runtime
            # validation sees the same contract the composer reports, without
            # mutating the caller-supplied config dict or changing node IDs for
            # unrelated fixed/flexible or explicit-guarantee cases.
            stored_config = dict(config)
            stored_config["schema"] = schema_config.to_dict()
        super().__init__(stored_config)

        self._path = cfg.resolved_path()
        self._column = cfg.column
        self._encoding = cfg.encoding
        self._strip_whitespace = cfg.strip_whitespace
        self._skip_blank_lines = cfg.skip_blank_lines
        self._schema_config = schema_config
        self._initialize_declared_guaranteed_fields(self._schema_config)
        self._on_validation_failure = cfg.on_validation_failure

        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "TextRowSchema",
            allow_coercion=True,
        )
        self.output_schema = self._schema_class

        initial_contract = create_contract_from_config(self._schema_config)
        if initial_contract.locked:
            self.set_schema_contract(initial_contract)
            self._contract_builder: ContractBuilder | None = None
        else:
            self._contract_builder = ContractBuilder(initial_contract)

        self._first_valid_row_processed = False

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        """Read the configured file line-by-line."""
        if not self._path.exists():
            raise FileNotFoundError(f"Text file not found: {self._path}")

        self._first_valid_row_processed = False
        line_num = 0
        skipped_blank_lines = 0

        try:
            with open(self._path, encoding=self._encoding, errors="surrogateescape", newline="") as f:
                for line_num, raw_line in enumerate(f, start=1):
                    if _contains_surrogateescape_chars(raw_line):
                        raw_bytes = _surrogateescape_line_to_bytes(raw_line, self._encoding)
                        quarantined = self._record_parse_error(
                            ctx=ctx,
                            row={
                                "file_path": str(self._path),
                                "__line_number__": line_num,
                                "__raw_bytes_hex__": raw_bytes.hex(),
                            },
                            error_msg=f"Text parse error at line {line_num}: invalid {self._encoding} encoding",
                        )
                        if quarantined is not None:
                            yield quarantined
                        continue

                    value = raw_line.rstrip("\r\n")
                    if self._strip_whitespace:
                        value = value.strip()

                    if self._skip_blank_lines and value == "":
                        skipped_blank_lines += 1
                        continue

                    # Shared composer/runtime contract helper depends on this:
                    # elspeth.contracts.schema.get_raw_producer_guaranteed_fields()
                    # infers {self._column} for observed text sources only.
                    # If you change which key the row uses, update that helper
                    # and its agreement tests.
                    yield from self._validate_and_yield(
                        {self._column: value},
                        ctx,
                    )
        except UnicodeDecodeError as exc:
            error_line = line_num + 1 if line_num > 0 else 1
            quarantined = self._record_parse_error(
                ctx=ctx,
                row={"file_path": str(self._path), "__line_number__": error_line},
                error_msg=f"Text parse error at line {error_line}: invalid {self._encoding} encoding ({exc})",
            )
            if quarantined is not None:
                yield quarantined

        if skipped_blank_lines > 0:
            # Audit the source-boundary filtering decision without inventing
            # synthetic quarantine rows for intentionally skipped blanks.
            ctx.record_validation_error(
                row={
                    "file_path": str(self._path),
                    "__skipped_blank_lines__": skipped_blank_lines,
                },
                error=f"Text source skipped {skipped_blank_lines} blank line(s) due to skip_blank_lines=True",
                schema_mode="parse",
                destination="discard",
            )

        if not self._first_valid_row_processed and self._contract_builder is not None:
            self.set_schema_contract(self._contract_builder.contract.with_locked())

    def _validate_and_yield(self, row: dict[str, Any], ctx: SourceContext) -> Iterator[SourceRow]:
        """Validate a line row and quarantine schema failures."""
        try:
            validated = self._schema_class.model_validate(row)
            validated_row = validated.to_row()

            if self._contract_builder is not None and not self._first_valid_row_processed:
                field_resolution = {self._column: self._column}
                self._contract_builder.process_first_row(validated_row, field_resolution)
                self.set_schema_contract(self._contract_builder.contract)
                self._first_valid_row_processed = True

            contract = self.require_schema_contract()
            if contract.locked:
                violations = contract.validate(validated_row)
                if violations:
                    error_msg = "; ".join(str(v) for v in violations)
                    ctx.record_validation_error(
                        row=validated_row,
                        error=error_msg,
                        schema_mode=self._schema_config.mode,
                        destination=self._on_validation_failure,
                    )
                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=validated_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    return

            yield SourceRow.valid(validated_row, contract=contract)
        except ValidationError as exc:
            ctx.record_validation_error(
                row=row,
                error=str(exc),
                schema_mode=self._schema_config.mode,
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=str(exc),
                    destination=self._on_validation_failure,
                )

    def _record_parse_error(
        self,
        ctx: SourceContext,
        row: Mapping[str, object],
        error_msg: str,
    ) -> SourceRow | None:
        """Record a parse error and quarantine unless discard mode."""
        row_payload = dict(row)
        ctx.record_validation_error(
            row=row_payload,
            error=error_msg,
            schema_mode="parse",
            destination=self._on_validation_failure,
        )
        if self._on_validation_failure == "discard":
            return None
        return SourceRow.quarantined(
            row=row_payload,
            error=error_msg,
            destination=self._on_validation_failure,
        )

    def close(self) -> None:
        """Release resources (no-op for file iteration source)."""
        pass
