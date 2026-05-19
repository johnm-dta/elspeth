"""Meta rule preventing new bespoke scripts/cicd/enforce_*.py gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from elspeth_lints.core.protocols import Category, Finding, RuleContext, RuleMetadata, RuleScope, Severity

RULE_ID = "meta.no-new-bespoke-cicd-enforcer"
MANIFEST_PATH = Path("config/cicd/lint_migration_status.yaml")
ACTIVE_STATUSES = frozenset({"pending", "shadow", "cutover"})
DELETED_STATUS = "deleted"
KNOWN_STATUSES = ACTIVE_STATUSES | frozenset({DELETED_STATUS})
RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="No new bespoke CI enforcers",
    description="New ELSPETH-specific CI checks must be elspeth-lints rules, not new scripts/cicd/enforce_*.py files.",
    severity=Severity.ERROR,
    category=Category.MANIFEST,
    cwe=(),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r"^scripts/cicd/enforce_.*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)


@dataclass(frozen=True, slots=True)
class NoNewBespokeCicdEnforcerRule:
    """Fail when an enforce_*.py script is not tracked by the migration manifest."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run the repository-scoped meta rule."""
        return self.analyze_repository(context.root, context)

    def analyze_repository(self, root: Path, context: RuleContext) -> list[Finding]:
        """Analyze the repository for unmanifested bespoke enforcer scripts."""
        del context
        findings: list[Finding] = []
        try:
            manifest = _load_manifest(root)
        except ValueError as exc:
            return [
                Finding(
                    rule_id=RULE_ID,
                    file_path=str(MANIFEST_PATH),
                    line=1,
                    column=0,
                    message=str(exc),
                    fingerprint="manifest-invalid",
                )
            ]

        actual_scripts = _actual_enforcer_scripts(root)
        actual_rel_paths = {path.relative_to(root).as_posix() for path in actual_scripts}

        for rel_path in sorted(actual_rel_paths.difference(manifest.active_paths)):
            findings.append(
                Finding(
                    rule_id=RULE_ID,
                    file_path=rel_path,
                    line=1,
                    column=0,
                    message=f"{rel_path} is not listed in {MANIFEST_PATH}; add an elspeth-lints rule instead of a bespoke script.",
                    fingerprint=f"unmanifested:{rel_path}",
                )
            )

        for rel_path in sorted(actual_rel_paths.intersection(manifest.deleted_paths)):
            findings.append(
                Finding(
                    rule_id=RULE_ID,
                    file_path=rel_path,
                    line=1,
                    column=0,
                    message=f"{rel_path} is marked deleted in {MANIFEST_PATH} but still exists.",
                    fingerprint=f"deleted-still-present:{rel_path}",
                )
            )

        for rel_path in sorted(manifest.active_paths.difference(actual_rel_paths)):
            findings.append(
                Finding(
                    rule_id=RULE_ID,
                    file_path=str(MANIFEST_PATH),
                    line=1,
                    column=0,
                    message=f"{rel_path} is active in {MANIFEST_PATH} but no matching file exists.",
                    fingerprint=f"manifest-stale:{rel_path}",
                )
            )

        return findings


@dataclass(frozen=True, slots=True)
class _MigrationManifest:
    active_paths: frozenset[str]
    deleted_paths: frozenset[str]


def _actual_enforcer_scripts(root: Path) -> tuple[Path, ...]:
    script_dir = root / "scripts/cicd"
    if not script_dir.exists():
        return ()
    return tuple(sorted(script_dir.glob("enforce_*.py")))


def _load_manifest(root: Path) -> _MigrationManifest:
    manifest_file = root / MANIFEST_PATH
    if not manifest_file.exists():
        return _MigrationManifest(active_paths=frozenset(), deleted_paths=frozenset())

    raw = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{MANIFEST_PATH} must be a YAML mapping")
    rules = _list_value(raw, "rules")

    active_paths: set[str] = set()
    deleted_paths: set[str] = set()
    for index, raw_rule in enumerate(rules):
        item = _mapping_value(raw_rule, f"rules[{index}]")
        old_script = _required_string(item, "old_script", context=f"rules[{index}]")
        status = _required_string(item, "status", context=f"rules[{index}]")
        if status not in KNOWN_STATUSES:
            expected = ", ".join(sorted(KNOWN_STATUSES))
            raise ValueError(f"rules[{index}].status must be one of: {expected}")
        if not old_script.startswith("scripts/cicd/enforce_") or not old_script.endswith(".py"):
            raise ValueError(f"rules[{index}].old_script must point at scripts/cicd/enforce_*.py")
        if status == DELETED_STATUS:
            deleted_paths.add(old_script)
        else:
            active_paths.add(old_script)

    return _MigrationManifest(active_paths=frozenset(active_paths), deleted_paths=frozenset(deleted_paths))


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


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    if key not in data:
        raise ValueError(f"{context} must include {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


RULE = NoNewBespokeCicdEnforcerRule()
