"""Core framework primitives for elspeth-lints."""

from __future__ import annotations

__all__ = [
    "Finding",
    "FindingKey",
    "RuleRegistry",
]

from elspeth_lints.core.allowlist import FindingKey
from elspeth_lints.core.findings import Finding
from elspeth_lints.core.registry import RuleRegistry
