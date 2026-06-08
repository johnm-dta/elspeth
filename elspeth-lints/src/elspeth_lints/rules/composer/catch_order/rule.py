"""Composer catch-order rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.composer.catch_order.metadata import LEGACY_RULE_ID, RULE_ID, RULE_METADATA, SUGGESTION

# The composer crash subclasses all descend from ComposerServiceError, which
# descends from Exception/BaseException — so a bare ``except Exception:`` (or
# ``BaseException``) before the narrow handler shadows it just as the named
# supertype does (elspeth-eb90341cdb).
_BROAD_SUPERTYPES: frozenset[str] = frozenset({"Exception", "BaseException"})
_SUBCLASS_TO_SUPERCLASSES: dict[str, frozenset[str]] = {
    "ComposerPluginCrashError": frozenset({"ComposerServiceError"}) | _BROAD_SUPERTYPES,
    "ComposerConvergenceError": frozenset({"ComposerServiceError"}) | _BROAD_SUPERTYPES,
    "ComposerRuntimePreflightError": frozenset({"ComposerServiceError"}) | _BROAD_SUPERTYPES,
    "_BadRequestLLMError": frozenset({"ComposerServiceError"}) | _BROAD_SUPERTYPES,
    "_MalformedLLMResponseError": frozenset({"ComposerServiceError"}) | _BROAD_SUPERTYPES,
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
    aliases = _build_exception_aliases(tree)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            findings.extend(_scan_try(node, file_path, aliases))
    return findings


def _build_exception_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local/import aliases of exception names back to the original name.

    Covers ``from mod import ComposerServiceError as CSE`` and module-level
    ``CSE = ComposerServiceError`` rebinds, so an aliased ``except CSE:`` resolves
    to the real supertype (elspeth-c0c4f49981).
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
        ):
            aliases[node.targets[0].id] = node.value.id
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
    return aliases


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    seen: set[str] = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def _scan_try(try_node: ast.Try, file_path: str, aliases: dict[str, str]) -> list[Finding]:
    handlers = [(handler, _handler_class_names(handler, aliases)) for handler in try_node.handlers]
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


def _handler_class_names(handler: ast.ExceptHandler, aliases: dict[str, str]) -> list[str]:
    if handler.type is None:
        return []
    nodes: list[ast.expr]
    if isinstance(handler.type, ast.Tuple):
        nodes = list(handler.type.elts)
    else:
        nodes = [handler.type]
    names: list[str] = []
    for node in nodes:
        name: str | None = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        if name is None:
            continue
        # Emit BOTH the literal name and the alias-resolved name (when they
        # differ) so an aliased handler matches without losing any literal match.
        names.append(name)
        resolved = _resolve_alias(name, aliases)
        if resolved != name:
            names.append(resolved)
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
