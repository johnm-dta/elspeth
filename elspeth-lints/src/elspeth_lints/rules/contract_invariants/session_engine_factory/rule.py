"""Session-engine factory rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, FindingKey, load_allowlist
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.contract_invariants.session_engine_factory.metadata import RULE_ID, RULE_METADATA, SUGGESTION
from elspeth_lints.rules.immutability.shared import allowlist_path_for_root

_ALLOWLIST_DIR = "enforce_session_engine_factory"


@dataclass(frozen=True, slots=True)
class SessionEngineFactoryRule:
    """Detect session database engines that bypass create_session_engine."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.INCREMENTAL
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one Python file for bare session create_engine calls."""
        display = _display_path(file_path, context.root)
        findings = find_session_engine_factory_findings(tree, display)
        return _apply_allowlist(
            findings,
            root=context.root,
            allowlist_dir_override=context.allowlist_dir_override,
        )


def find_session_engine_factory_findings(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return findings for bare SQLAlchemy create_engine calls in session contexts."""
    normalized = _normalize_source_path(file_path)
    if normalized == "web/sessions/engine.py":
        return []

    aliases = _sqlalchemy_create_engine_aliases(tree)
    if not aliases.names and not aliases.modules:
        return []

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_sqlalchemy_create_engine_call(node.func, aliases):
            continue
        reason = _violation_reason(normalized, node)
        if reason is None:
            continue
        findings.append(_finding(file_path=file_path, node=node, reason=reason))
    return findings


@dataclass(frozen=True, slots=True)
class _CreateEngineAliases:
    names: frozenset[str]
    modules: frozenset[str]


def _sqlalchemy_create_engine_aliases(tree: ast.AST) -> _CreateEngineAliases:
    names: set[str] = set()
    modules: set[str] = {"sqlalchemy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sqlalchemy":
            for alias in node.names:
                if alias.name == "create_engine":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlalchemy":
                    modules.add(alias.asname or "sqlalchemy")
    return _CreateEngineAliases(names=frozenset(names), modules=frozenset(modules))


def _is_sqlalchemy_create_engine_call(func: ast.expr, aliases: _CreateEngineAliases) -> bool:
    if isinstance(func, ast.Name):
        return func.id in aliases.names
    dotted = _dotted_name(func)
    if dotted is None:
        return False
    if dotted == "sqlalchemy.create_engine":
        return True
    return any(dotted == f"{module}.create_engine" for module in aliases.modules)


def _violation_reason(file_path: str, node: ast.Call) -> str | None:
    if _is_session_owned_path(file_path):
        return "session-owned path"
    if _is_web_path(file_path) and _call_mentions_session_db(node):
        return "session database URL"
    return None


def _is_session_owned_path(file_path: str) -> bool:
    return file_path.startswith("web/sessions/") or (file_path.startswith("tests/") and "/web/sessions/" in file_path)


def _is_web_path(file_path: str) -> bool:
    return file_path.startswith("web/") or (file_path.startswith("tests/") and "/web/" in file_path)


def _call_mentions_session_db(node: ast.Call) -> bool:
    expressions: list[ast.expr] = []
    if node.args:
        expressions.append(node.args[0])
    expressions.extend(keyword.value for keyword in node.keywords if keyword.arg == "url")
    return any(_expression_mentions_session_db(expression) for expression in expressions)


def _expression_mentions_session_db(expression: ast.expr) -> bool:
    try:
        rendered = ast.unparse(expression).lower()
    except Exception:
        rendered = ""
    if "session" not in rendered:
        return False
    return any(token in rendered for token in ("db", "database", "sqlite", "url"))


def _finding(*, file_path: str, node: ast.Call, reason: str) -> Finding:
    fingerprint_payload = (
        f"{RULE_ID}|{file_path}|{node.lineno}|{node.col_offset}|"
        f"{ast.dump(node.func, include_attributes=False)}|{_url_fingerprint_part(node)}"
    )
    return Finding(
        rule_id=RULE_ID,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=(f"Bare sqlalchemy.create_engine call in {file_path}:{node.lineno} bypasses create_session_engine for {reason}."),
        fingerprint=hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16],
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION,
    )


def _url_fingerprint_part(node: ast.Call) -> str:
    if node.args:
        return ast.dump(node.args[0], include_attributes=False)
    for keyword in node.keywords:
        if keyword.arg == "url":
            return ast.dump(keyword.value, include_attributes=False)
    return "_no_url_"


def _apply_allowlist(
    findings: list[Finding],
    *,
    root: Path,
    allowlist_dir_override: Path | None,
) -> list[Finding]:
    allowlist = _load_rule_allowlist(root=root, allowlist_dir_override=allowlist_dir_override)
    if allowlist is None:
        return findings
    return [
        finding
        for finding in findings
        if allowlist.match(
            FindingKey(
                file_path=finding.file_path,
                rule_id=finding.rule_id,
                symbol_context=(),
                fingerprint=finding.fingerprint,
            )
        )
        is None
    ]


def _load_rule_allowlist(*, root: Path, allowlist_dir_override: Path | None) -> Allowlist | None:
    allowlist_path = allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, _ALLOWLIST_DIR)
    if not allowlist_path.exists():
        return None
    return load_allowlist(allowlist_path, valid_rule_ids={RULE_ID})


def _normalize_source_path(file_path: str) -> str:
    if file_path.startswith("src/elspeth/"):
        return file_path.removeprefix("src/elspeth/")
    return file_path


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return node.attr
        return f"{parent}.{node.attr}"
    return None


def _display_path(file_path: Path, root: Path) -> str:
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return file_path.as_posix()


RULE = SessionEngineFactoryRule()
