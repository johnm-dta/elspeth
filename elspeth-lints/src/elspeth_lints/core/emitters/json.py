"""JSON emitter for parity and machine-readable local checks."""

from __future__ import annotations

import json
from collections.abc import Sequence

from elspeth_lints.core.protocols import Finding


def render_json(findings: Sequence[Finding]) -> str:
    """Render findings as a stable JSON list."""
    payload = [
        {
            "rule_id": finding.rule_id,
            "file_path": finding.file_path,
            "line": finding.line,
            "column": finding.column,
            "message": finding.message,
            "fingerprint": finding.fingerprint,
            "severity": finding.severity.value,
            "suggestion": finding.suggestion,
        }
        for finding in findings
    ]
    return json.dumps(payload, sort_keys=True) + "\n"
