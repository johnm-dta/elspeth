#!/usr/bin/env python3
"""CI lint: enforce title and description on plugin configuration fields.

Run from the project root:
    .venv/bin/python scripts/cicd/enforce_options_metadata.py

Allowlist entries live at config/cicd/enforce_options_metadata/allowlist.yaml.
Each entry has the form:
    {id: "<kind>/<plugin_name>:<field_name>", reason: "<why>"}

Discriminated variants use:
    <kind>/<plugin_name>[<variant>]:<field_name>
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager


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


def run_metadata_lint(*, plugin_manager: object, allowlist: set[str]) -> list[str]:
    """Return metadata failures for every plugin config field not allowlisted."""
    failures: list[str] = []
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
                    if not field_info.title:
                        failures.append(f"{identifier}: missing title")
                    if not field_info.description:
                        failures.append(f"{identifier}: missing description")
    return failures


def load_allowlist(path: Path) -> set[str]:
    """Load allowlisted identifiers and require reasons for every entry."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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


def _required_non_empty_string(entry: Mapping[str, object], key: str, *, path: Path, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: entries[{index}] must include non-empty {key!r}")
    return value


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    allowlist_path = root / "config" / "cicd" / "enforce_options_metadata" / "allowlist.yaml"
    try:
        allowlist = load_allowlist(allowlist_path)
    except OSError as exc:
        print(f"Plugin config metadata lint could not read allowlist: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Plugin config metadata lint has invalid allowlist: {exc}", file=sys.stderr)
        return 2

    failures = run_metadata_lint(plugin_manager=get_shared_plugin_manager(), allowlist=allowlist)
    if failures:
        print("Plugin config metadata lint failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nFix by adding title= and description= to each Field(...). "
            "Use the allowlist only for explicitly justified temporary exceptions.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
