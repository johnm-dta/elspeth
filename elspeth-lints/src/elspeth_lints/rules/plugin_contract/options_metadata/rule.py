"""Runtime-backed rule for plugin options metadata completeness."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import yaml
from pydantic import BaseModel

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.plugin_contract.options_metadata.metadata import ALLOWLIST_PATH, RULE_ID, RULE_METADATA


class PluginCatalog(Protocol):
    """Plugin-manager surface needed by this rule."""

    def get_sources(self) -> Iterable[object]:
        """Return source plugin classes."""

    def get_transforms(self) -> Iterable[object]:
        """Return transform plugin classes."""

    def get_sinks(self) -> Iterable[object]:
        """Return sink plugin classes."""


PluginManagerFactory = Callable[[], PluginCatalog]


def _default_plugin_manager() -> PluginCatalog:
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    return cast(PluginCatalog, get_shared_plugin_manager())


@dataclass(frozen=True, slots=True)
class FieldLocation:
    """Best-effort source location for one Pydantic model field."""

    file_path: str
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class OptionsMetadataRule:
    """Fail when plugin configuration fields lack title or description metadata."""

    plugin_manager_factory: PluginManagerFactory = _default_plugin_manager
    allowlist_path: Path = ALLOWLIST_PATH
    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run the repository-scoped metadata rule."""
        del tree, file_path
        allowlist = load_options_metadata_allowlist(context.root / self.allowlist_path)
        return collect_metadata_findings(
            plugin_manager=self.plugin_manager_factory(),
            allowlist=allowlist,
            root=context.root,
        )


def iter_metadata_models(kind: str, plugin_cls: object) -> Iterator[tuple[str, type[BaseModel]]]:
    """Yield the metadata-bearing config models for one plugin class."""
    plugin_name = getattr(plugin_cls, "name", None)
    if not isinstance(plugin_name, str):
        raise TypeError(f"{kind} plugin {plugin_cls!r} has no string name")
    variants_fn = getattr(plugin_cls, "discriminated_variants", None)
    if variants_fn is not None and callable(variants_fn):
        variants_raw = variants_fn()
        if not isinstance(variants_raw, tuple) or len(variants_raw) != 2:
            raise TypeError(f"{kind}/{plugin_name}: discriminated_variants() returned an invalid shape")
        variants = variants_raw[1]
        if not isinstance(variants, Mapping):
            raise TypeError(f"{kind}/{plugin_name}: discriminated variants must be a mapping")
        for variant_name, model in variants.items():
            if not isinstance(variant_name, str):
                raise TypeError(f"{kind}/{plugin_name}: discriminated variant names must be strings")
            if not isinstance(model, type) or not issubclass(model, BaseModel):
                raise TypeError(f"{kind}/{plugin_name}[{variant_name}]: variant config model must inherit BaseModel")
            yield f"{kind}/{plugin_name}[{variant_name}]", model
        return

    options_model = getattr(plugin_cls, "config_model", None)
    if isinstance(options_model, type) and issubclass(options_model, BaseModel):
        yield f"{kind}/{plugin_name}", options_model


def collect_metadata_findings(*, plugin_manager: PluginCatalog, allowlist: set[str], root: Path | None) -> list[Finding]:
    """Return metadata findings for every plugin config field not allowlisted."""
    findings: list[Finding] = []
    catalogs = (
        ("source", plugin_manager.get_sources()),
        ("transform", plugin_manager.get_transforms()),
        ("sink", plugin_manager.get_sinks()),
    )
    for kind, plugins in catalogs:
        for plugin_cls in plugins:
            for model_id, options_model in iter_metadata_models(kind, plugin_cls):
                for field_name, field_info in options_model.model_fields.items():
                    identifier = f"{model_id}:{field_name}"
                    if identifier in allowlist:
                        continue
                    location = _field_location(options_model, field_name, root=root)
                    if not field_info.title:
                        findings.append(_finding(identifier, "missing title", "missing-title", location))
                    if not field_info.description:
                        findings.append(_finding(identifier, "missing description", "missing-description", location))
    return findings


def load_options_metadata_allowlist(path: Path) -> set[str]:
    """Load legacy options-metadata allowlist identifiers and require reasons."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: allowlist YAML must be a mapping")
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'entries' must be a list")

    allowlist: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entries[{index}] must be a mapping")
        identifier = _required_non_empty_string(entry, "id", path=path, index=index)
        _required_non_empty_string(entry, "reason", path=path, index=index)
        allowlist.add(identifier)
    return allowlist


def _finding(identifier: str, message_suffix: str, fingerprint_suffix: str, location: FieldLocation) -> Finding:
    return Finding(
        rule_id=RULE_ID,
        file_path=location.file_path,
        line=location.line,
        column=location.column,
        message=f"{identifier}: {message_suffix}",
        fingerprint=f"{identifier}:{fingerprint_suffix}",
        severity=RULE_METADATA.severity,
    )


def _field_location(model: type[BaseModel], field_name: str, *, root: Path | None) -> FieldLocation:
    for cls in model.__mro__:
        if not isinstance(cls, type) or not issubclass(cls, BaseModel):
            continue
        location = _field_location_in_class(cls, field_name, root=root)
        if location is not None:
            return location
    return FieldLocation(file_path=f"plugin-config:{model.__name__}", line=1, column=0)


def _field_location_in_class(cls: type[BaseModel], field_name: str, *, root: Path | None) -> FieldLocation | None:
    source_file = inspect.getsourcefile(cls)
    if source_file is None:
        return None
    source_path = Path(source_file)
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError):
        return FieldLocation(file_path=_display_path(source_path, root=root), line=1, column=0)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            location = _field_location_in_class_node(node, field_name, source_path, root=root)
            if location is not None:
                return location
    return None


def _field_location_in_class_node(node: ast.ClassDef, field_name: str, source_path: Path, *, root: Path | None) -> FieldLocation | None:
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and _target_name(statement.target) == field_name:
            return FieldLocation(
                file_path=_display_path(source_path, root=root),
                line=statement.lineno,
                column=statement.col_offset,
            )
        if isinstance(statement, ast.Assign) and any(_target_name(target) == field_name for target in statement.targets):
            return FieldLocation(
                file_path=_display_path(source_path, root=root),
                line=statement.lineno,
                column=statement.col_offset,
            )
    return None


def _target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _display_path(source_path: Path, *, root: Path | None) -> str:
    if root is None:
        return str(source_path)
    try:
        return source_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(source_path)


def _required_non_empty_string(entry: Mapping[str, object], key: str, *, path: Path, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: entries[{index}] must include non-empty {key!r}")
    return value


RULE = OptionsMetadataRule()
