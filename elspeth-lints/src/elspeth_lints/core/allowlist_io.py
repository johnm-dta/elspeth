"""Shared YAML IO helpers for allowlist CI gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from elspeth_lints.core.allowlist import AllowlistEntry, _parse_allow_hits


class AllowlistIOError(RuntimeError):
    """An allowlist YAML document could not be read or parsed."""


@dataclass(frozen=True, slots=True)
class AllowlistYamlDocument:
    """One parsed allowlist YAML file."""

    source_file: str
    data: dict[str, Any]


def load_yaml_mapping_text(text: str, *, source_label: str) -> dict[str, Any]:
    """Parse YAML text as a mapping or raise ``AllowlistIOError``."""
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise AllowlistIOError(f"{source_label}: failed to parse as YAML mapping: {exc}") from exc
    if not isinstance(raw, dict):
        raise AllowlistIOError(f"{source_label}: failed to parse as YAML mapping: YAML root must be a mapping, got {type(raw).__name__}")
    return raw


def iter_yaml_documents(directory: Path) -> list[AllowlistYamlDocument]:
    """Return parsed non-default YAML files in ``directory``."""
    documents: list[AllowlistYamlDocument] = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        if yaml_file.name == "_defaults.yaml":
            continue
        try:
            text = yaml_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise AllowlistIOError(f"could not read {yaml_file}: {exc}") from exc
        documents.append(
            AllowlistYamlDocument(
                source_file=yaml_file.name,
                data=load_yaml_mapping_text(text, source_label=str(yaml_file)),
            )
        )
    return documents


def parse_allow_hits(data: dict[str, Any], *, source_file: str) -> list[AllowlistEntry]:
    """Parse ``allow_hits`` entries from one YAML mapping."""
    try:
        return _parse_allow_hits(data, source_file=source_file, source_root=None)
    except (ValueError, TypeError) as exc:
        raise AllowlistIOError(f"{source_file}: allow_hits entry shape violated loader invariants: {exc}") from exc


def iter_allow_hits_from_directory(directory: Path) -> list[AllowlistEntry]:
    """Return every ``allow_hits`` entry in a directory of allowlist YAML files."""
    entries: list[AllowlistEntry] = []
    for document in iter_yaml_documents(directory):
        if "allow_hits" not in document.data:
            continue
        entries.extend(parse_allow_hits(document.data, source_file=document.source_file))
    return entries


def entry_shape_count(data: dict[str, Any], key: str, *, source_file: str) -> int:
    """Return list length for an allowlist entry-shape key."""
    raw_entries = data.get(key, [])
    if raw_entries is None:
        return 0
    if not isinstance(raw_entries, list):
        raise AllowlistIOError(f"{source_file}: {key} must be a list if present")
    return len(raw_entries)
