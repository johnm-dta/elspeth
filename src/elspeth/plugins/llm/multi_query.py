"""Multi-query LLM support for case study x criteria cross-product evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QuerySpec:
    """Specification for a single query in the cross-product.

    Represents one (case_study, criterion) pair to be evaluated.

    Attributes:
        case_study_name: Name of the case study (e.g., "cs1")
        criterion_name: Name of the criterion (e.g., "diagnosis")
        input_fields: List of row field names to map to input_1, input_2, etc.
        output_prefix: Prefix for output fields (e.g., "cs1_diagnosis")
        criterion_data: Full criterion object for template injection
    """

    case_study_name: str
    criterion_name: str
    input_fields: list[str]
    output_prefix: str
    criterion_data: dict[str, Any]

    def build_template_context(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build template context for this query.

        Maps input_fields to input_1, input_2, etc. and injects criterion data.

        Args:
            row: Full row data

        Returns:
            Context dict with input_N, criterion, and row
        """
        context: dict[str, Any] = {}

        # Map input fields to positional variables
        # Access directly - missing field is a config error, should crash
        for i, field_name in enumerate(self.input_fields, start=1):
            if field_name not in row:
                raise KeyError(f"Required field '{field_name}' not found in row for query {self.output_prefix}")
            context[f"input_{i}"] = row[field_name]

        # Inject criterion data
        context["criterion"] = self.criterion_data

        # Include full row for row-based lookups
        context["row"] = row

        return context
