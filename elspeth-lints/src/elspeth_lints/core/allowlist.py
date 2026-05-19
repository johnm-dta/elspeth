"""Allowlist loading and matching for elspeth-lints."""

from __future__ import annotations

import fnmatch
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class FindingKey:
    """Stable key material for matching a finding against an allowlist."""

    file_path: str
    rule_id: str
    symbol_context: tuple[str, ...]
    fingerprint: str

    @property
    def canonical_key(self) -> str:
        """Return the exact suppression key."""
        symbol_part = ":".join(self.symbol_context) if self.symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:fp={self.fingerprint}"


@dataclass(slots=True)
class AllowlistEntry:
    """An exact allowlist entry for one finding fingerprint."""

    key: str
    owner: str
    reason: str
    safety: str
    expires: date | None
    file_fingerprint: str | None = None
    ast_path: str | None = None
    source_file: str = ""
    matched: bool = field(default=False, compare=False)

    def matches(self, finding: FindingKey) -> bool:
        """Return whether this exact entry suppresses the finding."""
        return self.key == finding.canonical_key


@dataclass(slots=True)
class PerFileRule:
    """A per-file allowlist rule for one or more rule ids."""

    pattern: str
    rules: tuple[str, ...]
    reason: str
    expires: date | None
    max_hits: int | None = None
    source_file: str = ""
    matched_count: int = field(default=0, compare=False)

    def matches(self, finding: FindingKey) -> bool:
        """Return whether this per-file rule suppresses the finding."""
        if finding.rule_id not in self.rules:
            return False
        return fnmatch.fnmatch(finding.file_path, self.pattern)


@dataclass(slots=True)
class Allowlist:
    """Loaded allowlist entries and per-file rules."""

    entries: list[AllowlistEntry]
    per_file_rules: list[PerFileRule] = field(default_factory=list)
    fail_on_stale: bool = True
    fail_on_expired: bool = True

    def match(self, finding: FindingKey) -> AllowlistEntry | PerFileRule | None:
        """Return the first matching suppression, if any."""
        for entry in self.entries:
            if entry.matches(finding):
                entry.matched = True
                return entry
        for rule in self.per_file_rules:
            if rule.matches(finding):
                rule.matched_count += 1
                return rule
        return None

    def get_unused_entries(self) -> list[AllowlistEntry]:
        """Return exact allowlist entries that did not match any finding."""
        return [entry for entry in self.entries if not entry.matched]

    def get_unused_rules(self) -> list[PerFileRule]:
        """Return per-file rules that did not match any finding."""
        return [rule for rule in self.per_file_rules if rule.matched_count == 0]

    def get_expired_entries(self) -> list[AllowlistEntry]:
        """Return exact allowlist entries whose expiry is in the past."""
        today = datetime.now(UTC).date()
        return [entry for entry in self.entries if entry.expires is not None and entry.expires < today]

    def get_expired_rules(self) -> list[PerFileRule]:
        """Return per-file allowlist rules whose expiry is in the past."""
        today = datetime.now(UTC).date()
        return [rule for rule in self.per_file_rules if rule.expires is not None and rule.expires < today]

    def get_exceeded_rules(self) -> list[PerFileRule]:
        """Return per-file allowlist rules that matched more than their cap."""
        return [rule for rule in self.per_file_rules if rule.max_hits is not None and rule.matched_count > rule.max_hits]


def load_allowlist(path: Path, *, valid_rule_ids: Collection[str]) -> Allowlist:
    """Load an allowlist from a YAML file or directory of YAML files."""
    if path.is_dir():
        defaults = _load_defaults(path / "_defaults.yaml")
        entries: list[AllowlistEntry] = []
        per_file_rules: list[PerFileRule] = []
        for yaml_file in sorted(file for file in path.glob("*.yaml") if file.name != "_defaults.yaml"):
            data = _load_yaml_file(yaml_file)
            entries.extend(_parse_allow_hits(data, source_file=yaml_file.name))
            per_file_rules.extend(_parse_per_file_rules(data, valid_rule_ids=valid_rule_ids, source_file=yaml_file.name))
        return Allowlist(
            entries=entries,
            per_file_rules=per_file_rules,
            fail_on_stale=defaults.fail_on_stale,
            fail_on_expired=defaults.fail_on_expired,
        )

    data = _load_yaml_file(path)
    defaults = _defaults_from_mapping(data)
    return Allowlist(
        entries=_parse_allow_hits(data, source_file=path.name),
        per_file_rules=_parse_per_file_rules(data, valid_rule_ids=valid_rule_ids, source_file=path.name),
        fail_on_stale=defaults.fail_on_stale,
        fail_on_expired=defaults.fail_on_expired,
    )


@dataclass(frozen=True, slots=True)
class _Defaults:
    fail_on_stale: bool = True
    fail_on_expired: bool = True


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: allowlist YAML must be a mapping")
    return raw


def _load_defaults(path: Path) -> _Defaults:
    if not path.exists():
        return _Defaults()
    return _defaults_from_mapping(_load_yaml_file(path))


def _defaults_from_mapping(data: dict[str, Any]) -> _Defaults:
    defaults = _mapping_or_empty(data, "defaults")
    return _Defaults(
        fail_on_stale=_bool_value(defaults, "fail_on_stale", default=True),
        fail_on_expired=_bool_value(defaults, "fail_on_expired", default=True),
    )


def _parse_allow_hits(data: dict[str, Any], *, source_file: str) -> list[AllowlistEntry]:
    entries_raw = _list_value(data, "allow_hits")
    entries: list[AllowlistEntry] = []
    for index, raw_entry in enumerate(entries_raw):
        entry = _mapping_value(raw_entry, f"allow_hits[{index}]")
        entries.append(
            AllowlistEntry(
                key=_required_string(entry, "key", context=f"allow_hits[{index}]"),
                owner=_required_string(entry, "owner", context=f"allow_hits[{index}]"),
                reason=_required_string(entry, "reason", context=f"allow_hits[{index}]"),
                safety=_required_string(entry, "safety", context=f"allow_hits[{index}]"),
                expires=_optional_date_alias(entry, "expires", "expires_at", context=f"allow_hits[{index}]"),
                file_fingerprint=_optional_string(entry, "file_fingerprint", context=f"allow_hits[{index}]"),
                ast_path=_optional_string(entry, "ast_path", context=f"allow_hits[{index}]"),
                source_file=source_file,
            )
        )
    return entries


def _parse_per_file_rules(data: dict[str, Any], *, valid_rule_ids: Collection[str], source_file: str) -> list[PerFileRule]:
    rules_raw = _list_value(data, "per_file_rules")
    rules: list[PerFileRule] = []
    for index, raw_rule in enumerate(rules_raw):
        item = _mapping_value(raw_rule, f"per_file_rules[{index}]")
        rule_ids = tuple(_string_list(item, "rules", context=f"per_file_rules[{index}]"))
        unknown = sorted(set(rule_ids).difference(valid_rule_ids))
        if unknown:
            raise ValueError(f"per_file_rules[{index}] uses unknown rule id(s): {', '.join(unknown)}")
        rules.append(
            PerFileRule(
                pattern=_required_string(item, "pattern", context=f"per_file_rules[{index}]"),
                rules=rule_ids,
                reason=_required_string(item, "reason", context=f"per_file_rules[{index}]"),
                expires=_optional_date(item, "expires", context=f"per_file_rules[{index}]"),
                max_hits=_optional_int(item, "max_hits", context=f"per_file_rules[{index}]"),
                source_file=source_file,
            )
        )
    return rules


def _mapping_or_empty(data: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in data:
        return {}
    return _mapping_value(data[key], key)


def _mapping_value(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _list_value(data: dict[str, Any], key: str) -> list[object]:
    if key not in data:
        return []
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _string_list(data: dict[str, Any], key: str, *, context: str) -> list[str]:
    raw = _list_value(data, key)
    values: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context}.{key}[{index}] must be a non-empty string")
        values.append(value)
    return values


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    if key not in data:
        raise ValueError(f"{context} must include {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_date(data: dict[str, Any], key: str, *, context: str) -> date | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be YYYY-MM-DD, null, or absent")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be YYYY-MM-DD") from exc


def _optional_date_alias(data: dict[str, Any], primary: str, alias: str, *, context: str) -> date | None:
    if primary in data and alias in data:
        raise ValueError(f"{context} must not include both {primary!r} and {alias!r}")
    if primary in data:
        return _optional_date(data, primary, context=context)
    return _optional_date(data, alias, context=context)


def _optional_string(data: dict[str, Any], key: str, *, context: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{context}.{key} must be a non-empty string, null, or absent")


def _optional_int(data: dict[str, Any], key: str, *, context: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, int):
        return value
    raise ValueError(f"{context}.{key} must be an integer, null, or absent")


def _bool_value(data: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"defaults.{key} must be a boolean")
    return value
