"""Composer catch-order rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.composer.catch_order.metadata import LEGACY_RULE_ID, RULE_ID, RULE_METADATA, SUGGESTION

_SUBCLASS_TO_SUPERCLASSES: dict[str, frozenset[str]] = {
    "ComposerPluginCrashError": frozenset({"ComposerServiceError"}),
    "ComposerConvergenceError": frozenset({"ComposerServiceError"}),
    "ComposerRuntimePreflightError": frozenset({"ComposerServiceError"}),
    "_BadRequestLLMError": frozenset({"ComposerServiceError"}),
    "_MalformedLLMResponseError": frozenset({"ComposerServiceError"}),
}


@dataclass(frozen=True, slots=True)
class ComposerCatchOrderRule:
    """Detect broad composer exception handlers before narrow handlers."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.INCREMENTAL
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one composer/web Python file."""
        return find_catch_order_findings(tree, display_path(file_path, context.root))


def find_catch_order_findings(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return CCO1 findings for one parsed file."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            findings.extend(_scan_try(node, file_path))
    return findings


def _scan_try(try_node: ast.Try, file_path: str) -> list[Finding]:
    handlers = [(handler, _handler_class_names(handler)) for handler in try_node.handlers]
    findings: list[Finding] = []
    for index, (handler_i, names_i) in enumerate(handlers):
        for handler_j, names_j in handlers[index + 1 :]:
            for sub_name in names_j:
                supertypes = _SUBCLASS_TO_SUPERCLASSES.get(sub_name)
                if not supertypes:
                    continue
                for super_name in names_i:
                    if super_name in supertypes:
                        findings.append(
                            _finding(
                                file_path=file_path, handler_i=handler_i, handler_j=handler_j, sub_name=sub_name, super_name=super_name
                            )
                        )
    return findings


def _handler_class_names(handler: ast.ExceptHandler) -> list[str]:
    if handler.type is None:
        return []
    nodes: list[ast.expr]
    if isinstance(handler.type, ast.Tuple):
        nodes = list(handler.type.elts)
    else:
        nodes = [handler.type]
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
    return names


def _finding(*, file_path: str, handler_i: ast.ExceptHandler, handler_j: ast.ExceptHandler, sub_name: str, super_name: str) -> Finding:
    fingerprint_payload = f"{LEGACY_RULE_ID}|{file_path}|{handler_j.lineno}|{sub_name}|{super_name}"
    return Finding(
        rule_id=LEGACY_RULE_ID,
        file_path=file_path,
        line=handler_j.lineno,
        column=0,
        message=(
            f"{file_path}:{handler_j.lineno}: except {sub_name} "
            f"is unreachable — preceding except {super_name} "
            f"at line {handler_i.lineno} catches the subclass. "
            "Move the narrower handler above the broader one."
        ),
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


RULE = ComposerCatchOrderRule()
