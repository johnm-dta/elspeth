# src/elspeth/core/landscape/formatters.py
"""Export formatters for Landscape data.

Formatters transform audit records for different output formats.
"""

import json
from typing import Any, Protocol


class ExportFormatter(Protocol):
    """Protocol for export formatters."""

    def format(self, record: dict[str, Any]) -> str | dict[str, Any]:
        """Format a record for output."""
        ...


class JSONFormatter:
    """Format records as JSON lines."""

    def format(self, record: dict[str, Any]) -> str:
        """Format as JSON line."""
        return json.dumps(record, default=str)


class CSVFormatter:
    """Format records for CSV output.

    Flattens nested structures using dot notation.
    """

    def flatten(self, record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dict to dot-notation keys."""
        result: dict[str, Any] = {}

        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self.flatten(value, full_key))
            elif isinstance(value, list):
                # Convert lists to JSON strings for CSV
                result[full_key] = json.dumps(value)
            else:
                result[full_key] = value

        return result

    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        """Format as flat dict for CSV."""
        return self.flatten(record)
