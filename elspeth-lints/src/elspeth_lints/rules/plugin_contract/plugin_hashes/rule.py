"""Plugin hash declaration rule implementation."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.plugin_contract.plugin_hashes.metadata import (
    RULE_ID,
    RULE_METADATA,
    RULE_PH1,
    RULE_PH2,
    RULE_PH3,
    SUGGESTION_PH1,
    SUGGESTION_PH2,
    SUGGESTION_PH3,
)

PLUGIN_DIRS = (
    "plugins/sources",
    "plugins/sinks",
    "plugins/transforms",
    "plugins/transforms/azure",
    "plugins/transforms/llm",
    "plugins/transforms/rag",
)
EXCLUDED_FILES = frozenset(
    {
        "__init__.py",
        "base.py",
        "config.py",
        "validation.py",
        "templates.py",
        "langfuse.py",
        "tracing.py",
        "multi_query.py",
        "capacity_errors.py",
        "provider.py",
    }
)
EXPECTED_PLUGIN_COUNT = 37
_HASH_LINE_PATTERN = re.compile(rb'(\s*source_file_hash\s*(?::[^=]+=\s*|=\s*))"sha256:[^"]+"')
_NORMALIZED_HASH_VALUE = b'"sha256:0000000000000000"'


@dataclass(frozen=True, slots=True)
class PluginAttributes:
    """Extracted plugin class declarations."""

    class_name: str
    plugin_version: str | None
    source_file_hash: str | None


@dataclass(frozen=True, slots=True)
class PluginHashesRule:
    """Detect missing or stale plugin version/hash declarations."""

    min_plugins: int = EXPECTED_PLUGIN_COUNT
    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run the repository-scoped plugin-hash rule."""
        del tree, file_path
        return scan_root(context.root, min_plugins=self.min_plugins)


def scan_root(root: Path, *, min_plugins: int = EXPECTED_PLUGIN_COUNT) -> list[Finding]:
    """Scan plugin files for PH1/PH2/PH3 findings."""
    plugin_files = discover_plugin_files(root)
    plugin_count = sum(len(extract_plugin_attributes(path)) for path in plugin_files)
    if plugin_count < min_plugins:
        return [
            Finding(
                rule_id=RULE_ID,
                file_path=".",
                line=1,
                column=0,
                message=f"DISCOVERY ERROR: found {plugin_count} plugins, expected at least {min_plugins}. Check --root path and PLUGIN_DIRS.",
                fingerprint=f"discovery:{plugin_count}:{min_plugins}",
                severity=RULE_METADATA.severity,
            )
        ]

    findings: list[Finding] = []
    for path in plugin_files:
        relative_path = display_path(path, root)
        attrs_list = extract_plugin_attributes(path)
        if not attrs_list:
            continue
        computed_hash = compute_source_file_hash(path)
        for attrs in attrs_list:
            if attrs.plugin_version is None or attrs.plugin_version == "0.0.0":
                findings.append(
                    _finding(
                        rule_id=RULE_PH1,
                        file_path=relative_path,
                        class_name=attrs.class_name,
                        message=f"{relative_path} ({attrs.class_name}): no version declaration (plugin_version is {attrs.plugin_version!r})",
                        suggestion=SUGGESTION_PH1,
                    )
                )
            if attrs.source_file_hash is None:
                findings.append(
                    _finding(
                        rule_id=RULE_PH2,
                        file_path=relative_path,
                        class_name=attrs.class_name,
                        message=f"{relative_path} ({attrs.class_name}): no source_file_hash declaration",
                        suggestion=SUGGESTION_PH2,
                    )
                )
            elif attrs.source_file_hash != computed_hash:
                findings.append(
                    _finding(
                        rule_id=RULE_PH3,
                        file_path=relative_path,
                        class_name=attrs.class_name,
                        message=(
                            f"{relative_path} ({attrs.class_name}): stale source_file_hash\n"
                            f"  declared: {attrs.source_file_hash}\n"
                            f"  expected: {computed_hash}"
                        ),
                        suggestion=SUGGESTION_PH3,
                    )
                )
    return findings


def discover_plugin_files(root: Path) -> list[Path]:
    """Find plugin entry-point files under the legacy plugin directories."""
    files: list[Path] = []
    for relative_dir in PLUGIN_DIRS:
        directory = root / relative_dir
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_") or path.name in EXCLUDED_FILES:
                continue
            files.append(path)
    return files


def compute_source_file_hash(file_path: Path) -> str:
    """Compute the legacy plugin source hash with self-referential normalization."""
    raw = file_path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized.startswith(b"\xef\xbb\xbf"):
        normalized = normalized[3:]
    normalized = _HASH_LINE_PATTERN.sub(lambda match: match.group(1) + _NORMALIZED_HASH_VALUE, normalized)
    digest = hashlib.sha256(normalized).hexdigest()[:16]
    return f"sha256:{digest}"


def extract_plugin_attributes(file_path: Path) -> list[PluginAttributes]:
    """Extract plugin class declarations from one source file."""
    source = file_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(file_path))
    module_string_constants = _module_string_constants(tree)
    results: list[PluginAttributes] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _has_name_class_attribute(node, module_string_constants):
            continue
        version_value, _ = _get_class_attribute_value(node, "plugin_version", module_string_constants)
        hash_value, _ = _get_class_attribute_value(node, "source_file_hash", module_string_constants)
        results.append(
            PluginAttributes(
                class_name=node.name,
                plugin_version=version_value if isinstance(version_value, str) else None,
                source_file_hash=hash_value if isinstance(hash_value, str) else None,
            )
        )
    return results


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Return top-level names bound directly to string constants."""
    constants: dict[str, str] = {}
    for item in tree.body:
        if isinstance(item, ast.Assign) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = item.value.value
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            constants[item.target.id] = item.value.value
    return constants


def _get_class_attribute_value(
    node: ast.ClassDef,
    attr_name: str,
    module_string_constants: dict[str, str] | None = None,
) -> tuple[object | None, int | None]:
    sentinel = object()
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    if isinstance(item.value, ast.Constant):
                        return (item.value.value, item.lineno)
                    if isinstance(item.value, ast.Name) and module_string_constants is not None:
                        resolved = module_string_constants.get(item.value.id, sentinel)
                        if resolved is not sentinel:
                            return (resolved, item.lineno)
                    return (sentinel, item.lineno)
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == attr_name:
            if item.value is not None and isinstance(item.value, ast.Constant):
                return (item.value.value, item.lineno)
            if item.value is not None and isinstance(item.value, ast.Name) and module_string_constants is not None:
                resolved = module_string_constants.get(item.value.id, sentinel)
                if resolved is not sentinel:
                    return (resolved, item.lineno)
            if item.value is None:
                return (None, None)
            return (sentinel, item.lineno)
    return (None, None)


def _has_name_class_attribute(node: ast.ClassDef, module_string_constants: dict[str, str]) -> bool:
    value, _ = _get_class_attribute_value(node, "name", module_string_constants)
    return isinstance(value, str)


def _finding(*, rule_id: str, file_path: str, class_name: str, message: str, suggestion: str) -> Finding:
    fingerprint_payload = f"{rule_id}|{file_path}|{class_name}"
    return Finding(
        rule_id=rule_id,
        file_path=file_path,
        line=1,
        column=0,
        message=message,
        fingerprint=hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16],
        severity=RULE_METADATA.severity,
        suggestion=suggestion,
    )


def display_path(path: Path, root: Path) -> str:
    """Return a path relative to the scan root when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


RULE = PluginHashesRule()
