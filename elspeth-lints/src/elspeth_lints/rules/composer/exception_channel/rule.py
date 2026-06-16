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
    aliases = _exception_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        name = _raise_exception_name(node.exc, aliases)
        if name in _BANNED:
            findings.append(_finding(file_path=file_path, line=node.lineno, name=name))
    return findings


def _raise_exception_name(exc: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(exc, ast.Call):
        return _exception_reference_name(exc.func, aliases)
    return _exception_reference_name(exc, aliases)


def _exception_aliases(tree: ast.AST) -> dict[str, str]:
    """Return simple aliases to banned builtins exception classes."""
    aliases: dict[str, str] = {"builtins": "builtins"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    aliases[alias.asname or alias.name] = "builtins"
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for alias in node.names:
                if alias.name in _BANNED:
                    aliases[alias.asname or alias.name] = alias.name
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            resolved = _exception_reference_name(value, aliases)
            if resolved not in _BANNED:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = resolved
    return aliases


def _exception_reference_name(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Name):
        return _canonical_exception_name(aliases.get(expr.id, expr.id))
    if isinstance(expr, ast.Attribute):
        dotted = _dotted_name(expr)
        if dotted is None:
            return None
        parts = dotted.split(".")
        if parts[0] in aliases:
            dotted = ".".join((aliases[parts[0]], *parts[1:]))
        return _canonical_exception_name(dotted)
    return None


def _dotted_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted_name(expr.value)
        if base is None:
            return None
        return f"{base}.{expr.attr}"
    return None


def _canonical_exception_name(name: str) -> str | None:
    if name in _BANNED:
        return name
    if name.startswith("builtins."):
        candidate = name.rsplit(".", 1)[-1]
        if candidate in _BANNED:
            return candidate
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
