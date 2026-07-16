"""Metadata for the portable SQLite insert-builder rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "contract_invariants.portable_sqlite_insert"
SUGGESTION = (
    "Pair sqlalchemy.dialects.sqlite.insert with sqlalchemy.dialects.postgresql.insert and dispatch explicitly "
    "on connection.dialect.name for both 'sqlite' and 'postgresql'."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Portable SQLite insert builder",
    description="SQLite-specific insert builders require an explicit PostgreSQL counterpart and dialect dispatch.",
    severity=Severity.ERROR,
    category=Category.CONTRACT_INVARIANTS,
    cwe=("CWE-754",),
    scope=RuleScope.INCREMENTAL,
    path_filter=r".*\.py$",
    examples_violation_count=2,
    examples_clean_count=3,
)
