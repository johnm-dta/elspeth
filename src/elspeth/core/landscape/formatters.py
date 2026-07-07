"""Export formatters for Landscape data.

Formatters transform audit records for different output formats.
"""

import json
from collections.abc import Mapping
from typing import Any

from elspeth.core.landscape.lineage_text import LineageTextFormatter as LineageTextFormatter
from elspeth.core.landscape.serialization import dataclass_to_dict as dataclass_to_dict
from elspeth.core.landscape.serialization import serialize_datetime


class JSONFormatter:
    """Format records as JSON lines."""

    def format(self, record: dict[str, Any]) -> str:
        """Format as JSON line."""
        normalized = serialize_datetime(record)
        return json.dumps(normalized, allow_nan=False)


class CSVFormatter:
    """Format records for CSV output.

    Flattens nested structures using dot notation.
    """

    def flatten(self, record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested mappings to dot-notation keys.

        Raises:
            ValueError: If flattened keys collide (e.g., dotted source keys
                vs nested mapping keys produce identical paths).
        """
        result: dict[str, Any] = {}

        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, Mapping):
                if not value:
                    # Preserve empty mappings as JSON "{}" — an empty object is a
                    # distinct datum from absence. Auditors must be able to tell
                    # "config was explicitly empty" from "config was not present".
                    if full_key in result:
                        raise ValueError(f"CSV flatten key collision: '{full_key}' already exists. Audit export would lose data.")
                    result[full_key] = "{}"
                    continue
                nested = self.flatten(dict(value.items()), full_key)
                for nested_key, nested_val in nested.items():
                    if nested_key in result:
                        raise ValueError(
                            f"CSV flatten key collision: '{nested_key}' produced by both a nested mapping and a prior key. "
                            "Audit export would lose data."
                        )
                    result[nested_key] = nested_val
            elif isinstance(value, list):
                if full_key in result:
                    raise ValueError(f"CSV flatten key collision: '{full_key}' already exists. Audit export would lose data.")
                # Convert lists to JSON strings for CSV
                # Use serialize_datetime to validate (rejects NaN/Infinity) and convert datetimes
                result[full_key] = json.dumps(serialize_datetime(value))
            else:
                if full_key in result:
                    raise ValueError(f"CSV flatten key collision: '{full_key}' already exists. Audit export would lose data.")
                # Validate scalar values and normalize datetimes to ISO strings.
                result[full_key] = serialize_datetime(value)

        return result

    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        """Format as flat dict for CSV."""
        return self.flatten(record)
