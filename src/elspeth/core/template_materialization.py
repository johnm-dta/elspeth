"""Load-time materialization for file-backed plugin template options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

PluginCollectionName = Literal["transforms", "aggregations"]
ContentKind = Literal["text", "yaml"]

PLUGIN_OPTION_COLLECTIONS: tuple[PluginCollectionName, ...] = ("transforms", "aggregations")


class TemplateFileError(Exception):
    """Error loading template, lookup, or system prompt files."""


@dataclass(frozen=True, slots=True)
class FileBackedTemplateOption:
    file_key: str
    content_key: str
    source_key: str
    label: str
    content_kind: ContentKind


FILE_BACKED_TEMPLATE_OPTION_REGISTRY: tuple[FileBackedTemplateOption, ...] = (
    FileBackedTemplateOption(
        file_key="template_file",
        content_key="prompt_template",
        source_key="prompt_template_source",
        label="Template file",
        content_kind="text",
    ),
    FileBackedTemplateOption(
        file_key="lookup_file",
        content_key="lookup",
        source_key="lookup_source",
        label="Lookup file",
        content_kind="yaml",
    ),
    FileBackedTemplateOption(
        file_key="system_prompt_file",
        content_key="system_prompt",
        source_key="system_prompt_source",
        label="System prompt file",
        content_kind="text",
    ),
)
FILE_BACKED_TEMPLATE_OPTION_KEYS = frozenset(rule.file_key for rule in FILE_BACKED_TEMPLATE_OPTION_REGISTRY)


def _resolve_template_path(file_ref: str, settings_path: Path, label: str) -> Path:
    """Resolve a template/lookup/prompt file path with containment check."""
    config_root = settings_path.parent.resolve()
    file_path = Path(file_ref)
    if not file_path.is_absolute():
        file_path = (config_root / file_path).resolve()
    else:
        file_path = file_path.resolve()

    try:
        file_path.relative_to(config_root)
    except ValueError as exc:
        raise TemplateFileError(
            f"{label} path traversal blocked: {file_ref!r} resolves to {file_path} which is outside config directory {config_root}"
        ) from exc

    if not file_path.exists():
        raise TemplateFileError(f"{label} not found: {file_path}")

    return file_path


class TemplateOptionMaterializer:
    """Materialize file-backed plugin options using a single option registry."""

    def __init__(self, settings_path: Path) -> None:
        self._settings_path = settings_path

    def materialize_config(self, raw_config: Mapping[str, Any]) -> dict[str, Any]:
        config = dict(raw_config)
        for collection_name in PLUGIN_OPTION_COLLECTIONS:
            collection = config.get(collection_name)
            if not isinstance(collection, list):
                continue
            config[collection_name] = [self._materialize_plugin_config(plugin_config) for plugin_config in collection]
        return config

    def materialize_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(options)
        for rule in FILE_BACKED_TEMPLATE_OPTION_REGISTRY:
            if rule.file_key not in result:
                continue
            if rule.content_key in result:
                raise TemplateFileError(f"Cannot specify both '{rule.content_key}' and '{rule.file_key}'")
            file_ref = result.pop(rule.file_key)
            file_path = _resolve_template_path(file_ref, self._settings_path, rule.label)
            result[rule.content_key] = self._load_content(rule, file_path)
            result[rule.source_key] = file_ref
        return result

    @staticmethod
    def reject_file_backed_options(raw_config: Mapping[str, object]) -> None:
        for collection_name in PLUGIN_OPTION_COLLECTIONS:
            collection = raw_config[collection_name] if collection_name in raw_config else None
            if type(collection) is not list:
                continue
            for index, plugin_config in enumerate(collection):
                if type(plugin_config) is not dict:
                    continue
                options = plugin_config["options"] if "options" in plugin_config else None
                if type(options) is not dict:
                    continue
                present = sorted(key for key in FILE_BACKED_TEMPLATE_OPTION_KEYS if key in options)
                if not present:
                    continue
                raw_name = plugin_config["name"] if "name" in plugin_config else index
                raise ValueError(
                    "load_settings_from_yaml_string() cannot expand file-backed template options "
                    f"{present} for {collection_name}[{raw_name!r}] because in-memory web execution "
                    "has no trusted settings file base path. Use load_settings() for file-backed "
                    "configs, or inline prompt_template, lookup, and system_prompt before web validation/execution."
                )

    def _materialize_plugin_config(self, plugin_config: Any) -> Any:
        if not isinstance(plugin_config, dict):
            return plugin_config
        plugin = dict(plugin_config)
        options = plugin.get("options")
        if isinstance(options, dict):
            plugin["options"] = self.materialize_options(options)
        return plugin

    def _load_content(self, rule: FileBackedTemplateOption, file_path: Path) -> Any:
        if rule.content_kind == "text":
            return file_path.read_text(encoding="utf-8")
        try:
            loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            return loaded if loaded is not None else {}
        except yaml.YAMLError as e:
            raise TemplateFileError(f"Invalid YAML in lookup file: {e}") from e


def _expand_template_files(options: dict[str, Any], settings_path: Path) -> dict[str, Any]:
    return TemplateOptionMaterializer(settings_path).materialize_options(options)
