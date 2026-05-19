"""SARIF 2.1.0 emitter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from elspeth_lints.core.protocols import Finding, RuleMetadata, Severity

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
TOOL_NAME = "elspeth-lints"
TOOL_URI = "https://github.com/johnm-dta/elspeth/tree/main/elspeth-lints"


def render_sarif(findings: Sequence[Finding], *, metadata: Iterable[RuleMetadata] = ()) -> str:
    """Render findings as SARIF 2.1.0."""
    metadata_by_id = {item.id: item for item in metadata}
    for finding in findings:
        if finding.rule_id not in metadata_by_id:
            metadata_by_id[finding.rule_id] = _fallback_metadata(finding)

    payload: dict[str, Any] = {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": TOOL_NAME,
                        "informationUri": TOOL_URI,
                        "rules": [_rule_payload(metadata_by_id[rule_id]) for rule_id in sorted(metadata_by_id)],
                    }
                },
                "results": [_result_payload(finding) for finding in findings],
            }
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _rule_payload(metadata: RuleMetadata) -> dict[str, Any]:
    return {
        "id": metadata.id,
        "name": metadata.name,
        "shortDescription": {"text": metadata.description},
        "fullDescription": {"text": metadata.description},
        "help": {"text": metadata.description},
        "properties": {
            "category": metadata.category.value,
            "precision": "high",
            "severity": metadata.severity.value,
            "tags": list(metadata.cwe),
        },
    }


def _result_payload(finding: Finding) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ruleId": finding.rule_id,
        "level": _sarif_level(finding.severity),
        "message": {"text": finding.message},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": finding.file_path},
                    "region": {
                        "startLine": max(finding.line, 1),
                        "startColumn": max(finding.column + 1, 1),
                    },
                }
            }
        ],
        "fingerprints": {"elspeth-lints/v1": finding.fingerprint},
        "partialFingerprints": {"primaryLocationLineHash": finding.fingerprint},
    }
    if finding.suggestion is not None:
        result["properties"] = {"suggestion": finding.suggestion}
    return result


def _sarif_level(severity: Severity) -> str:
    if severity is Severity.WARNING:
        return "warning"
    if severity is Severity.NOTE:
        return "note"
    return "error"


def _fallback_metadata(finding: Finding) -> RuleMetadata:
    from elspeth_lints.core.protocols import Category, RuleScope

    return RuleMetadata(
        id=finding.rule_id,
        name=finding.rule_id,
        description=finding.rule_id,
        severity=finding.severity,
        category=Category.MANIFEST,
        cwe=(),
        scope=RuleScope.INCREMENTAL,
        path_filter=r".*",
        examples_violation_count=0,
        examples_clean_count=0,
    )
