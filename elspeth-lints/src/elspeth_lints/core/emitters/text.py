"""Human-readable text emitter."""

from __future__ import annotations

from collections.abc import Sequence

from elspeth_lints.core.protocols import Finding


def render_text(findings: Sequence[Finding]) -> str:
    """Render findings for terminal output."""
    return "".join(f"{finding.file_path}:{finding.line}:{finding.column}: {finding.rule_id}: {finding.message}\n" for finding in findings)
