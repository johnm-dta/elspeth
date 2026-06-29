"""Tier-1 decoration rule implementation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import (
    ParsedPythonFile,
    PythonFileReadError,
    PythonSyntaxError,
    iter_python_files,
    parse_python_file,
)
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.audit_evidence.shared import (
    allowlist_path_for_root,
    class_allowlist_governance_findings_for_root,
    display_path,
    load_class_allowlist,
    repo_relative_display_path,
)
from elspeth_lints.rules.audit_evidence.tier_1_decoration.metadata import (
    RULE_ID,
    RULE_METADATA,
    RULE_TDE1,
    RULE_TDE2,
    SUGGESTION_TDE1,
    SUGGESTION_TDE2,
)

_CHECKED_SUFFIXES = ("Error", "Violation")
_TIER_1_ERROR_MODULES = frozenset({"elspeth.contracts.errors", "elspeth.contracts.tier_registry"})


@dataclass(frozen=True, slots=True)
class _Tier1ErrorBindings:
    direct_names: frozenset[str]
    module_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class Tier1DecorationRule:
    """Detect missing Tier-1/Tier-2 exception class declarations."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one syntax tree for tests, or scan the configured errors.py."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return scan_tree(tree, display_path(file_path, context.root), _source_lines(file_path))
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Scan the legacy errors.py target, or fixture roots that lack it."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_tier_1_decoration")
    )
    allowlist = load_class_allowlist(allowlist_dir)
    candidates = _scan_candidates(root)
    candidate_set = {path.resolve() for path in candidates}
    findings: list[Finding] = []
    # TDE1 (class decoration) is scoped to the canonical errors.py target — the
    # codebase intentionally decorates only that central module. TDE2 (the
    # caller_module spoofing guard) applies to EVERY tier_1_error call site
    # repo-wide (elspeth-f8650893f1): scan the candidates fully, then make a
    # TDE2-only pass over all other files.
    for path in candidates:
        parsed = _parse_or_raise(path)
        display = repo_relative_display_path(path, root) if path.name == "errors.py" else display_path(path, root)
        findings.extend(scan_tree(parsed.tree, display, parsed.source.splitlines()))
    # iter_python_files applies the shared excluded-dir filter (.venv, .uv-cache,
    # .git, build, dist, node_modules, __pycache__, ...). A raw rglob here would
    # descend into a vendored tree and let _parse_or_raise turn one third-party
    # file (BOM prefix, py2 syntax) into a hard crash of the whole gate.
    for path in iter_python_files(root):
        if path.resolve() in candidate_set:
            continue
        parsed = _parse_or_raise(path)
        findings.extend(scan_tree(parsed.tree, display_path(path, root), parsed.source.splitlines(), emit_tde1=False))
    active = [finding for finding in findings if finding.rule_id != RULE_TDE1 or allowlist.match_key(finding.fingerprint) is None]
    return [
        *active,
        *class_allowlist_governance_findings_for_root(
            allowlist,
            allowlist_dir,
            root=root,
            allowlist_dir_override=allowlist_dir_override,
        ),
    ]


def _parse_or_raise(path: Path) -> ParsedPythonFile:
    parsed = parse_python_file(path)
    if isinstance(parsed, PythonSyntaxError):
        raise SyntaxError(f"{parsed.path}:{parsed.line}:{parsed.column}: {parsed.message}")
    if isinstance(parsed, PythonFileReadError):
        # A read error after enumeration indicates a race between filesystem
        # walk and parse — be loud rather than silently skip, matching the
        # syntax-error policy.
        raise OSError(f"{parsed.path}: {parsed.message}")
    return parsed


def scan_tree(tree: ast.AST, file_path: str, source_lines: list[str], *, emit_tde1: bool = True) -> list[Finding]:
    """Return TDE1/TDE2 findings for one parsed syntax tree.

    ``emit_tde1=False`` restricts the scan to TDE2 (the caller_module spoofing
    guard), used for files outside the canonical errors.py target where the
    class-decoration requirement (TDE1) does not apply.
    """
    findings: list[Finding] = []
    bindings = _tier_1_error_bindings(tree)
    for node in ast.walk(tree):
        if emit_tde1 and isinstance(node, ast.ClassDef):
            finding = _tde1_finding(file_path, node, source_lines, bindings)
            if finding is not None:
                findings.append(finding)
                continue
        if isinstance(node, ast.Call) and _canonical_tier_1_error_call(node, bindings):
            detail = _caller_module_violation(node)
            if detail is not None:
                findings.append(_tde2_finding(file_path, node, detail))
    return findings


def _scan_candidates(root: Path) -> list[Path]:
    repo_target = root / "src" / "elspeth" / "contracts" / "errors.py"
    if repo_target.exists():
        return [repo_target]
    source_target = root / "contracts" / "errors.py"
    if source_target.exists():
        return [source_target]
    # Same excluded-dir discipline as the TDE2 pass: never raw-rglob a tree that
    # may contain a vendored / cache subdir whose files do not parse.
    return list(iter_python_files(root))


def _tde1_finding(file_path: str, node: ast.ClassDef, source_lines: list[str], bindings: _Tier1ErrorBindings) -> Finding | None:
    if not node.name.endswith(_CHECKED_SUFFIXES):
        return None
    if _has_tier_1_error_decorator(node, bindings):
        return None
    if _has_tier_2_comment(node, source_lines):
        return None
    key = f"{file_path}:{RULE_TDE1}:{node.name}"
    return Finding(
        rule_id=RULE_TDE1,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=f"{node.name} has no @tier_1_error(reason=...) decorator and no # TIER-2: justification comment",
        fingerprint=key,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION_TDE1,
    )


def _tde2_finding(file_path: str, node: ast.Call, detail: str) -> Finding:
    key = f"{file_path}:{RULE_TDE2}:{node.lineno}"
    return Finding(
        rule_id=RULE_TDE2,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=detail,
        fingerprint=key,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION_TDE2,
    )


def _has_tier_1_error_decorator(class_node: ast.ClassDef, bindings: _Tier1ErrorBindings) -> bool:
    return any(
        isinstance(decorator, ast.Call) and _canonical_tier_1_error_call(decorator, bindings) and _has_nonempty_reason(decorator)
        for decorator in class_node.decorator_list
    )


def _tier_1_error_bindings(tree: ast.AST) -> _Tier1ErrorBindings:
    direct_names: set[str] = set()
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _TIER_1_ERROR_MODULES:
            for alias in node.names:
                if alias.name == "tier_1_error":
                    direct_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _TIER_1_ERROR_MODULES:
                    module_names.add(alias.asname or alias.name)
    return _Tier1ErrorBindings(frozenset(direct_names), frozenset(module_names))


def _canonical_tier_1_error_call(call: ast.Call, bindings: _Tier1ErrorBindings) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in bindings.direct_names
    if isinstance(func, ast.Attribute) and func.attr == "tier_1_error":
        return _dotted_name(func.value) in bindings.module_names
    return False


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base is not None:
            return f"{base}.{node.attr}"
    return None


def _has_nonempty_reason(call: ast.Call) -> bool:
    """True when the tier_1_error call carries a non-empty ``reason``.

    ``reason`` is a required keyword on the runtime decorator; a missing or
    empty ``reason`` is rejected at runtime, so a decorator without one does not
    satisfy the Tier-1 decoration requirement (elspeth-08b0336287). A non-literal
    reason cannot be proven empty statically and is treated as present.
    """
    for keyword in call.keywords:
        if keyword.arg is None:
            # **kwargs splat — cannot prove reason is missing/empty.
            return True
        if keyword.arg == "reason":
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return bool(value.value.strip())
            return True
    return False


def _has_tier_2_comment(class_node: ast.ClassDef, source_lines: list[str]) -> bool:
    decorator_lines: set[int] = set()
    for decorator in class_node.decorator_list:
        for lineno in range(decorator.lineno, (decorator.end_lineno or decorator.lineno) + 1):
            decorator_lines.add(lineno)

    idx = class_node.lineno - 2
    while idx >= 0:
        line = source_lines[idx] if idx < len(source_lines) else ""
        stripped = line.strip()
        if not stripped or (idx + 1) in decorator_lines:
            idx -= 1
            continue
        marker = "# TIER-2:"
        if marker not in line:
            return False
        return bool(line[line.index(marker) + len(marker) :].strip())
    return False


def _caller_module_violation(call: ast.Call) -> str | None:
    caller_module_kw: ast.keyword | None = None
    for keyword in call.keywords:
        if keyword.arg == "caller_module":
            caller_module_kw = keyword
            break
    if caller_module_kw is None:
        return "missing caller_module kwarg (require caller_module=__name__)"
    value = caller_module_kw.value
    if not (isinstance(value, ast.Name) and value.id == "__name__"):
        return f"caller_module value must be the __name__ literal, got {ast.dump(value)}"
    return None


def _source_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


RULE = Tier1DecorationRule()
