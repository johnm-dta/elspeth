# src/elspeth/plugins/sources/csv_source.py
"""CSV source plugin for ELSPETH.

Loads rows from CSV files using csv.reader for proper multiline quoted field support.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.
"""

import contextlib
import csv
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.contracts.contract_builder import ContractBuilder
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import TabularSourceDataConfig
from elspeth.plugins.schema_factory import create_schema_from_config
from elspeth.plugins.sources.field_normalization import FieldResolution, resolve_field_names


class CSVSourceConfig(TabularSourceDataConfig):
    """Configuration for CSV source plugin.

    Inherits from TabularSourceDataConfig, which provides:
    - schema and on_validation_failure (from SourceDataConfig)
    - columns, normalize_fields, field_mapping (field normalization)
    """

    delimiter: str = ","
    encoding: str = "utf-8"
    skip_rows: int = 0


class CSVSource(BaseSource):
    """Load rows from a CSV file.

    Config options:
        path: Path to CSV file (required)
        schema: Schema configuration (required, via SourceDataConfig)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
        skip_rows: Number of header rows to skip (default: 0)

    Field normalization options (via TabularSourceDataConfig):
        normalize_fields: Normalize messy headers to valid identifiers (default: False)
        field_mapping: Override specific normalized names (requires normalize_fields)
        columns: Explicit column names for headerless files (mutually exclusive
                 with normalize_fields)

    The schema can be:
        - Observed: {"mode": "observed"} - accept any fields
        - Fixed: {"mode": "fixed", "fields": ["id: int", "name: str"]}
        - Flexible: {"mode": "flexible", "fields": ["id: int"]} - at least these fields
    """

    name = "csv"
    plugin_version = "1.0.0"
    # Override parent type - SourceDataConfig requires this to be set
    _on_validation_failure: str

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSourceConfig.from_dict(config)

        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding
        self._skip_rows = cfg.skip_rows

        # Store normalization config for use in load()
        self._columns = cfg.columns
        self._normalize_fields = cfg.normalize_fields
        self._field_mapping = cfg.field_mapping

        # Field resolution computed at load() time - includes version for audit
        self._field_resolution: FieldResolution | None = None

        # Store schema config for audit trail (required by DataPluginConfig)
        self._schema_config = cfg.schema_config

        # Store quarantine routing destination
        self._on_validation_failure = cfg.on_validation_failure
        # on_success is injected by the instantiation bridge (cli_helpers.py)

        # CRITICAL: allow_coercion=True for sources (external data boundary)
        # Sources are the ONLY place where type coercion is allowed
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "CSVRowSchema",
            allow_coercion=True,
        )

        # Set output_schema for protocol compliance
        self.output_schema = self._schema_class

        # Create initial schema contract (may be updated after first row)
        # Contract creation deferred until load() when field_resolution is known
        self._contract_builder: ContractBuilder | None = None

    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load rows from CSV file with optional field normalization.

        Uses csv.reader directly on file handle to properly support
        multiline quoted fields (e.g., "field with\nembedded newline").

        Field resolution modes:
        - Default: Headers read from file, used as-is
        - normalize_fields=True: Headers normalized to valid Python identifiers
        - columns=[...]: Headerless file, use explicit column names

        Each row is validated against the configured schema:
        - Valid rows are yielded as SourceRow.valid()
        - Invalid rows are yielded as SourceRow.quarantined()

        Yields:
            SourceRow for each row (valid or quarantined).

        Raises:
            FileNotFoundError: If CSV file does not exist.
            ValueError: If field collision detected after normalization,
                       or column count mismatch in headerless mode.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        # CRITICAL: newline='' required for proper embedded newline handling
        # See: https://docs.python.org/3/library/csv.html
        with open(self._path, encoding=self._encoding, newline="") as f:
            # Create csv.reader on file handle for multiline field support
            reader = csv.reader(f, delimiter=self._delimiter)

            # Skip CSV records as configured (not raw lines), preserving multiline alignment.
            # csv.Error is caught because skip_rows targets non-CSV metadata preamble
            # (comments, version headers, etc.) that may contain unmatched quotes or
            # other constructs invalid under RFC 4180.  The user explicitly asked to
            # discard these rows, so a parse failure is not an error worth surfacing.
            for _ in range(self._skip_rows):
                with contextlib.suppress(csv.Error):
                    next(reader, None)

            # Determine headers based on config
            if self._columns is not None:
                # Headerless mode - use explicit columns
                raw_headers = None
            else:
                # Read header row from file
                try:
                    raw_headers = next(reader)
                except StopIteration:
                    return  # Empty file after skip_rows
                except csv.Error as e:
                    # Header parse failure at source boundary (Tier 3): record and quarantine/discard
                    physical_line = reader.line_num if reader.line_num > 0 else self._skip_rows + 1
                    raw_row = {
                        "file_path": str(self._path),
                        "__line_number__": physical_line,
                        "__raw_line__": "(unparseable CSV header)",
                    }
                    error_msg = f"CSV parse error at line {physical_line}: {e}"

                    ctx.record_validation_error(
                        row=raw_row,
                        error=error_msg,
                        schema_mode="parse",
                        destination=self._on_validation_failure,
                    )

                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=raw_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    return

            # Resolve field names (normalization + mapping)
            # This may raise ValueError on collision
            self._field_resolution = resolve_field_names(
                raw_headers=raw_headers,
                normalize_fields=self._normalize_fields,
                field_mapping=self._field_mapping,
                columns=self._columns,
            )
            headers = self._field_resolution.final_headers
            expected_count = len(headers)

            # Create initial contract with field resolution
            initial_contract = create_contract_from_config(
                self._schema_config,
                field_resolution=self._field_resolution.resolution_mapping,
            )
            self._contract_builder = ContractBuilder(initial_contract)

            # Track whether first valid row has been processed (for type inference)
            first_valid_row_processed = False

            # Process data rows with manual iteration to catch csv.Error per row
            row_num = 0  # Logical row number (data rows only)
            while True:
                try:
                    # Try to read next row - csv.Error raised here for malformed rows
                    values = next(reader)
                except StopIteration:
                    break  # End of file
                except csv.Error as e:
                    # CSV parsing error (bad quoting, unmatched quotes, etc.)
                    # Quarantine this row instead of crashing the run
                    row_num += 1
                    physical_line = reader.line_num
                    raw_row = {
                        "__raw_line__": "(unparseable due to csv.Error)",
                        "__line_number__": physical_line,
                        "__row_number__": row_num,
                    }
                    error_msg = f"CSV parse error at line {physical_line}: {e}"

                    ctx.record_validation_error(
                        row=raw_row,
                        error=error_msg,
                        schema_mode="parse",
                        destination=self._on_validation_failure,
                    )

                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=raw_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    continue  # Skip to next row

                # Skip empty rows (blank lines in CSV)
                # csv.reader returns [] for blank lines, which would cause field count mismatch
                if not values:
                    continue

                row_num += 1
                # reader.line_num tracks physical file line position (including multiline fields)
                physical_line = reader.line_num

                # Column count validation - quarantine malformed rows in both header and headerless modes
                # Per Three-Tier Trust Model: source data is Tier 3 (zero trust), quarantine bad rows
                if len(values) != expected_count:
                    raw_row = {
                        "__raw_line__": self._delimiter.join(values),
                        "__line_number__": physical_line,
                        "__row_number__": row_num,
                    }
                    error_msg = f"CSV parse error at line {physical_line}: expected {expected_count} fields, got {len(values)}"

                    ctx.record_validation_error(
                        row=raw_row,
                        error=error_msg,
                        schema_mode="parse",
                        destination=self._on_validation_failure,
                    )

                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=raw_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    continue

                # Build row dict
                row = dict(zip(headers, values, strict=False))

                # Validate row against schema
                try:
                    validated = self._schema_class.model_validate(row)
                    validated_row = validated.to_row()

                    # Process first valid row for type inference
                    if not first_valid_row_processed:
                        self._contract_builder.process_first_row(
                            validated_row,
                            self._field_resolution.resolution_mapping,
                        )
                        self.set_schema_contract(self._contract_builder.contract)
                        first_valid_row_processed = True

                    # Validate against locked contract to catch type drift on
                    # inferred fields. Pydantic extra="allow" accepts any type
                    # for extras — the contract enforces inferred types here.
                    contract = self.get_schema_contract()
                    if contract is not None and contract.locked:
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
                            continue

                    yield SourceRow.valid(
                        validated_row,
                        contract=contract,
                    )
                except ValidationError as e:
                    ctx.record_validation_error(
                        row=row,
                        error=str(e),
                        schema_mode=self._schema_config.mode,
                        destination=self._on_validation_failure,
                    )

                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=row,
                            error=str(e),
                            destination=self._on_validation_failure,
                        )

            # CRITICAL: Handle empty source case (all rows quarantined or no rows)
            # If no valid rows were processed, the contract is still unlocked.
            # Lock it now so downstream consumers have a consistent contract state.
            if not first_valid_row_processed and self._contract_builder is not None:
                self.set_schema_contract(self._contract_builder.contract.with_locked())

    def close(self) -> None:
        """Release resources (no-op for CSV source)."""
        pass

    def get_field_resolution(self) -> tuple[dict[str, str], str | None] | None:
        """Return field resolution mapping for audit trail.

        Returns the mapping from original CSV headers to final field names,
        computed during load() when normalize_fields or field_mapping is used.

        Returns:
            Tuple of (resolution_mapping, normalization_version) if field resolution
            was computed, or None if load() hasn't been called yet or no normalization
            was needed.
        """
        if self._field_resolution is None:
            return None

        return (
            self._field_resolution.resolution_mapping,
            self._field_resolution.normalization_version,
        )
