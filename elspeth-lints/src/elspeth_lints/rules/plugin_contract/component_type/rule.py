"""Plugin component-type rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, FindingKey, load_allowlist
from elspeth_lints.core.allowlist_governance import allowlist_governance_findings_for_root
from elspeth_lints.core.ast_walker import PythonFileReadError, PythonSyntaxError, walk_python_files
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.plugin_contract.component_type.metadata import LEGACY_RULE_ID, RULE_ID, RULE_METADATA, SUGGESTION

_ALL_RULE_IDS = frozenset({LEGACY_RULE_ID})
_KNOWN_BASES: dict[str, dict[str, bool]] = {
    "DataPluginConfig": {"is_data_config_descendant": True, "sets_type": False, "is_exempt": True},
    "PathConfig": {"is_data_config_descendant": True, "sets_type": False, "is_exempt": True},
    "SourceDataConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
    "SinkPathConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
    "TransformDataConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
    "TabularSourceDataConfig": {"is_data_config_descendant": True, "sets_type": True, "is_exempt": False},
}


@dataclass(frozen=True, slots=True)
class ClassInfo:
    """Information about a class discovered during AST scanning."""

    name: str
    file_path: str
    line: int
    column: int
    base_names: list[str]
    sets_component_type: bool
    is_exempt: bool
    code_snippet: str


@dataclass(frozen=True, slots=True)
class ComponentTypeRule:
    """Detect DataPluginConfig descendants that do not declare a component type."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run a whole-root scan, or a direct tree scan for focused tests."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return find_component_type_findings(scan_tree_classes(tree, display_path(file_path, context.root), []))
        return scan_root(
            context.root,
            allowlist_dir_override=context.allowlist_dir_override,
            governance_emitted_dirs=context.allowlist_governance_emitted_dirs,
            emit_allowlist_governance=context.emit_allowlist_governance,
        )


def scan_root(
    root: Path,
    *,
    allowlist_dir_override: Path | None = None,
    governance_emitted_dirs: set[str] | None = None,
    emit_allowlist_governance: bool = True,
) -> list[Finding]:
    """Scan a root and apply the legacy component-type allowlist."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_component_type")
    )
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_ALL_RULE_IDS)
    all_classes: list[ClassInfo] = []
    for parsed in walk_python_files(root):
        if isinstance(parsed, (PythonSyntaxError, PythonFileReadError)):
            continue
        source_lines = parsed.source.splitlines()
        all_classes.extend(scan_tree_classes(parsed.tree, display_path(parsed.path, root), source_lines))
    findings = find_component_type_findings(all_classes)
    active = [finding for finding in findings if _allowlist_match(allowlist, finding) is None]
    return [
        *active,
        *allowlist_governance_findings_for_root(
            allowlist,
            allowlist_dir,
            root=root,
            allowlist_dir_override=allowlist_dir_override,
            emitted_dirs=governance_emitted_dirs,
            enabled=emit_allowlist_governance,
        ),
    ]


def scan_tree_classes(tree: ast.AST, file_path: str, source_lines: list[str]) -> list[ClassInfo]:
    """Extract class information from one AST."""
    classes: list[ClassInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        snippet = source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else f"class {node.name}"
        classes.append(
            ClassInfo(
                name=node.name,
                file_path=file_path,
                line=node.lineno,
                column=node.col_offset,
                base_names=_extract_base_names(node),
                sets_component_type=_class_sets_component_type(node),
                is_exempt=_class_is_exempt(node),
                code_snippet=snippet,
            )
        )
    return classes


def find_component_type_findings(all_classes: list[ClassInfo]) -> list[Finding]:
    """Resolve class inheritance and return CT1 findings."""
    registry = {info.name: info for info in all_classes}
    descendant_cache: dict[str, bool] = {}
    ancestor_cache: dict[str, bool | None] = {}
    findings: list[Finding] = []

    for info in all_classes:
        if not _is_checked_descendant(info, registry, descendant_cache):
            continue
        if info.sets_component_type or info.is_exempt:
            continue
        if _ancestor_sets_type(info.name, registry, ancestor_cache):
            continue

        fingerprint_payload = f"{LEGACY_RULE_ID}|{info.file_path}|{info.name}"
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
        findings.append(
            Finding(
                rule_id=LEGACY_RULE_ID,
                file_path=info.file_path,
                line=info.line,
                column=info.column,
                message=(
                    f"{info.name} inherits from DataPluginConfig but does not set "
                    "_plugin_component_type to 'source', 'sink', or 'transform'"
                ),
                fingerprint=fingerprint,
                severity=RULE_METADATA.severity,
                suggestion=SUGGESTION,
            )
        )
    return findings


def _extract_base_names(node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _class_sets_component_type(node: ast.ClassDef) -> bool:
    for item in node.body:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == "_plugin_component_type"
            and item.value is not None
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            return True
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_plugin_component_type"
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    return True
    return False


def _class_is_exempt(node: ast.ClassDef) -> bool:
    for item in node.body:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == "_component_type_exempt"
            and item.value is not None
            and isinstance(item.value, ast.Constant)
            and item.value.value is True
        ):
            return True
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_component_type_exempt"
                    and isinstance(item.value, ast.Constant)
                    and item.value.value is True
                ):
                    return True
    return False


def _is_checked_descendant(info: ClassInfo, registry: dict[str, ClassInfo], cache: dict[str, bool]) -> bool:
    return any(_is_data_config_descendant(base, registry, cache) for base in info.base_names) or any(
        base in _KNOWN_BASES and _KNOWN_BASES[base]["is_data_config_descendant"] for base in info.base_names
    )


def _is_data_config_descendant(class_name: str, registry: dict[str, ClassInfo], cache: dict[str, bool]) -> bool:
    if class_name in cache:
        return cache[class_name]
    if class_name in _KNOWN_BASES:
        result = _KNOWN_BASES[class_name]["is_data_config_descendant"]
        cache[class_name] = result
        return result
    info = registry.get(class_name)
    if info is None:
        cache[class_name] = False
        return False
    cache[class_name] = False
    for base in info.base_names:
        if _is_data_config_descendant(base, registry, cache):
            cache[class_name] = True
            return True
    return False


def _ancestor_sets_type(class_name: str, registry: dict[str, ClassInfo], cache: dict[str, bool | None]) -> bool:
    if class_name in cache:
        result = cache[class_name]
        return result if result is not None else False
    if class_name in _KNOWN_BASES:
        result = _KNOWN_BASES[class_name]["sets_type"]
        cache[class_name] = result
        return result
    info = registry.get(class_name)
    if info is None:
        cache[class_name] = False
        return False

    cache[class_name] = None
    for base in info.base_names:
        base_info = registry.get(base)
        base_sets = _KNOWN_BASES[base]["sets_type"] if base in _KNOWN_BASES else bool(base_info and base_info.sets_component_type)
        if base_sets or _ancestor_sets_type(base, registry, cache):
            cache[class_name] = True
            return True
    cache[class_name] = False
    return False


def _allowlist_match(allowlist: Allowlist, finding: Finding) -> object | None:
    return allowlist.match(
        FindingKey(
            file_path=finding.file_path,
            rule_id=finding.rule_id,
            symbol_context=(finding.message.split(" ", 1)[0],),
            fingerprint=finding.fingerprint,
        )
    )


def display_path(file_path: Path, root: Path) -> str:
    """Return a path relative to the scan root when possible."""
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def allowlist_path_for_root(root: Path, directory_name: str) -> Path:
    """Find a CI allowlist directory from a scan root or repository cwd."""
    relative = Path("config") / "cicd" / directory_name
    candidates = [root, Path.cwd(), *root.parents]
    for candidate in candidates:
        path = candidate / relative
        if path.exists():
            return path
    return root / relative


RULE = ComponentTypeRule()
