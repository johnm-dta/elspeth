"""Shared commencement gate expression contract."""

from __future__ import annotations

from elspeth.core.expression_parser import ExpressionParser

COMMENCEMENT_GATE_ALLOWED_NAMES = ("collections", "dependency_runs", "env")


def validate_commencement_gate_condition(condition: str) -> None:
    """Validate a commencement gate condition against the shared context contract."""
    ExpressionParser(condition, allowed_names=list(COMMENCEMENT_GATE_ALLOWED_NAMES))
