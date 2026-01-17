# src/elspeth/plugins/sources/csv_source.py
"""CSV source plugin for ELSPETH.

Loads rows from CSV files using pandas for robust parsing.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.
"""

from collections.abc import Iterator
from typing import Any

import pandas as pd
from pydantic import ValidationError

from elspeth.contracts import PluginSchema
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import SourceDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class CSVSourceConfig(SourceDataConfig):
    """Configuration for CSV source plugin.

    Inherits from SourceDataConfig, which requires schema and on_validation_failure.
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

    The schema can be:
        - Dynamic: {"fields": "dynamic"} - accept any fields
        - Strict: {"mode": "strict", "fields": ["id: int", "name: str"]}
        - Free: {"mode": "free", "fields": ["id: int"]} - at least these fields
    """

    name = "csv"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSourceConfig.from_dict(config)

        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding
        self._skip_rows = cfg.skip_rows
        self._dataframe: pd.DataFrame | None = None

        # Store schema config for audit trail
        # SourceDataConfig (via DataPluginConfig) ensures schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # Store quarantine routing destination
        self._on_validation_failure = cfg.on_validation_failure

        # CRITICAL: allow_coercion=True for sources (external data boundary)
        # Sources are the ONLY place where type coercion is allowed
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "CSVRowSchema",
            allow_coercion=True,
        )

        # Set output_schema for protocol compliance
        self.output_schema = self._schema_class

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load rows from CSV file.

        Each row is validated against the configured schema:
        - Valid rows are yielded for processing
        - Invalid rows are quarantined (recorded but not yielded)

        Yields:
            Dict for each valid row with column names as keys.

        Raises:
            FileNotFoundError: If CSV file does not exist.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        self._dataframe = pd.read_csv(
            self._path,
            delimiter=self._delimiter,
            encoding=self._encoding,
            skiprows=self._skip_rows,
            dtype=str,  # Keep all values as strings for consistent handling
            keep_default_na=False,  # Don't convert empty strings to NaN
        )

        # DataFrame columns are strings from CSV headers
        for record in self._dataframe.to_dict(orient="records"):
            row = {str(k): v for k, v in record.items()}

            try:
                # Validate and potentially coerce row data
                validated = self._schema_class.model_validate(row)
                yield validated.to_row()
            except ValidationError as e:
                # Record validation failure - row quarantined
                # This is a trust boundary: external data may be invalid
                ctx.record_validation_error(
                    row=row,
                    error=str(e),
                    schema_mode=self._schema_config.mode or "dynamic",
                    destination=self._on_validation_failure,
                )

                # Route to quarantine sink if not discarding
                if self._on_validation_failure != "discard":
                    ctx.route_to_sink(
                        sink_name=self._on_validation_failure,
                        row=row,
                        metadata={"validation_error": str(e)},
                    )
                # Skip invalid row - don't yield
                continue

    def close(self) -> None:
        """Release resources."""
        self._dataframe = None
