"""Composer exception-channel rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.composer.exception_channel.metadata import LEGACY_RULE_ID, RULE_ID, RULE_METADATA, SUGGESTION

_BANNED = frozenset({"TypeError", "ValueError", "UnicodeError", "UnicodeDecodeError", "UnicodeEncodeError"})


@dataclass(frozen=True, slots=True)
class ComposerExceptionChannelRule:
    """Detect bare TypeError/ValueError/UnicodeError raises in composer tools."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.INCREMENTAL
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one composer tool file."""
        return find_exception_channel_findings(tree, display_path(file_path, context.root))


def find_exception_channel_findings(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return CEC1 findings for one parsed file."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        name = _raise_exception_name(node.exc)
        if name in _BANNED:
            findings.append(_finding(file_path=file_path, line=node.lineno, name=name))
    return findings


def _raise_exception_name(exc: ast.expr) -> str | None:
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id
    if isinstance(exc, ast.Name):
        return exc.id
    return None


def _finding(*, file_path: str, line: int, name: str) -> Finding:
    fingerprint_payload = f"{LEGACY_RULE_ID}|{file_path}|{line}|{name}"
    return Finding(
        rule_id=LEGACY_RULE_ID,
        file_path=file_path,
        line=line,
        column=0,
        message=f"raise {name}(...) at {file_path}:{line} — use ToolArgumentError",
        fingerprint=hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16],
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION,
    )


def display_path(file_path: Path, root: Path) -> str:
    """Return a path relative to root when possible."""
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return file_path.as_posix()


RULE = ComposerExceptionChannelRule()
