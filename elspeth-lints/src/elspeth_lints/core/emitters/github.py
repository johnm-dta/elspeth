"""GitHub Actions workflow-command emitter."""

from __future__ import annotations

from collections.abc import Sequence

from elspeth_lints.core.protocols import Finding


def render_github(findings: Sequence[Finding]) -> str:
    """Render findings as GitHub Actions annotations."""
    return "".join(_render_finding(finding) for finding in findings)


def _render_finding(finding: Finding) -> str:
    level = "warning" if finding.severity.value == "warning" else "notice" if finding.severity.value == "note" else "error"
    return (
        f"::{level} file={_escape_property(finding.file_path)},"
        f"line={max(finding.line, 1)},"
        f"col={max(finding.column + 1, 1)},"
        f"title={_escape_property(finding.rule_id)}::"
        f"{_escape_message(finding.message)}\n"
    )


def _escape_property(value: str) -> str:
    return _escape_message(value).replace(",", "%2C").replace(":", "%3A")


def _escape_message(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
