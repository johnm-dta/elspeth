"""Metadata for the session-engine factory rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "contract_invariants.session_engine_factory"
SUGGESTION = (
    "Use elspeth.web.sessions.engine.create_session_engine() for session database engines so SQLite "
    "foreign-key, WAL, busy-timeout, and startup probes cannot be bypassed."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Session engine factory",
    description="Session database engines must be built through create_session_engine, not bare sqlalchemy.create_engine.",
    severity=Severity.ERROR,
    category=Category.CONTRACT_INVARIANTS,
    cwe=("CWE-754",),
    scope=RuleScope.INCREMENTAL,
    path_filter=r"(^|/)(src/elspeth/web/|web/|tests/(unit|integration|property|e2e)/web/).+\.py$",
    examples_violation_count=2,
    examples_clean_count=3,
)
