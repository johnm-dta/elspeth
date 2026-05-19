#!/usr/bin/env python3
"""Compare legacy CICD enforcer findings with elspeth-lints rule findings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MANIFEST = Path("config/cicd/lint_migration_status.yaml")
SUCCESS_CODES = frozenset({0, 1})


@dataclass(frozen=True, slots=True)
class MigrationRule:
    """One legacy-script to elspeth-lints-rule migration entry."""

    old_script: str
    new_rule: str | None
    status: str
    old_command: tuple[str, ...] | None = None
    new_command: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class MigrationManifest:
    """Parsed lint migration manifest."""

    version: int
    rules: tuple[MigrationRule, ...]

    def shadow_rules(self) -> tuple[MigrationRule, ...]:
        """Return rules that must pass old-vs-new parity."""
        return tuple(rule for rule in self.rules if rule.status == "shadow")


@dataclass(frozen=True, order=True, slots=True)
class NormalizedFinding:
    """Comparable representation shared by old scripts and elspeth-lints."""

    file_path: str
    line: int
    column: int
    rule_id: str
    fingerprint: str
    message: str


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured command output."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ParityComparison:
    """Parity result for one manifest entry."""

    old_script: str
    new_rule: str
    old_command: tuple[str, ...]
    new_command: tuple[str, ...]
    missing_from_new: list[NormalizedFinding]
    unexpected_in_new: list[NormalizedFinding]
    old_stderr: str = ""
    new_stderr: str = ""

    @property
    def ok(self) -> bool:
        """Return whether the old and new finding sets match exactly."""
        return not self.missing_from_new and not self.unexpected_in_new


@dataclass(frozen=True, slots=True)
class ParityRunResult:
    """Aggregate parity-harness result."""

    comparisons: list[ParityComparison]

    @property
    def ok(self) -> bool:
        """Return whether every comparison passed."""
        return all(comparison.ok for comparison in self.comparisons)


def load_manifest(path: Path) -> MigrationManifest:
    """Load the lint migration manifest."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: manifest must be a YAML mapping")
    version = _required_int(raw, "version", context=str(path))
    rules_raw = _list_value(raw, "rules", context=str(path))
    rules: list[MigrationRule] = []
    for index, raw_rule in enumerate(rules_raw):
        item = _mapping_value(raw_rule, f"rules[{index}]")
        rules.append(
            MigrationRule(
                old_script=_required_string(item, "old_script", context=f"rules[{index}]"),
                new_rule=_optional_string(item, "new_rule", context=f"rules[{index}]"),
                status=_required_string(item, "status", context=f"rules[{index}]"),
                old_command=_optional_string_tuple(item, "old_command", context=f"rules[{index}]"),
                new_command=_optional_string_tuple(item, "new_command", context=f"rules[{index}]"),
            )
        )
    return MigrationManifest(version=version, rules=tuple(rules))


def run_parity(manifest: MigrationManifest, *, root: Path) -> ParityRunResult:
    """Run all shadow parity comparisons in a manifest."""
    comparisons = [compare_rule(rule, root=root) for rule in manifest.shadow_rules()]
    return ParityRunResult(comparisons=comparisons)


def compare_rule(rule: MigrationRule, *, root: Path) -> ParityComparison:
    """Run and compare one legacy/new rule pair."""
    if rule.new_rule is None:
        raise ValueError(f"{rule.old_script}: shadow entries must include new_rule")
    old_command = _expanded_command(_old_command_template(rule), rule=rule, root=root)
    new_command = _expanded_command(_new_command_template(rule), rule=rule, root=root)

    old_result = _run_command(old_command, root=root)
    new_result = _run_command(new_command, root=root)
    old_findings = set(_normalize_findings(old_result.stdout, source=f"{rule.old_script} stdout"))
    new_findings = set(_normalize_findings(new_result.stdout, source=f"{rule.new_rule} stdout"))

    return ParityComparison(
        old_script=rule.old_script,
        new_rule=rule.new_rule,
        old_command=old_command,
        new_command=new_command,
        missing_from_new=sorted(old_findings.difference(new_findings)),
        unexpected_in_new=sorted(new_findings.difference(old_findings)),
        old_stderr=old_result.stderr,
        new_stderr=new_result.stderr,
    )


def render_text(result: ParityRunResult) -> str:
    """Render a human-readable parity report."""
    if not result.comparisons:
        return "No shadow migration entries configured.\n"
    lines: list[str] = []
    for comparison in result.comparisons:
        if comparison.ok:
            lines.append(f"PASS {comparison.old_script} -> {comparison.new_rule}")
            continue
        lines.append(f"FAIL {comparison.old_script} -> {comparison.new_rule}")
        for finding in comparison.missing_from_new:
            lines.append(f"  missing_from_new: {_finding_text(finding)}")
        for finding in comparison.unexpected_in_new:
            lines.append(f"  unexpected_in_new: {_finding_text(finding)}")
    return "\n".join(lines) + "\n"


def render_json(result: ParityRunResult) -> str:
    """Render a structured JSON parity report."""
    payload = {
        "ok": result.ok,
        "comparisons": [
            {
                "old_script": comparison.old_script,
                "new_rule": comparison.new_rule,
                "old_command": list(comparison.old_command),
                "new_command": list(comparison.new_command),
                "missing_from_new": [asdict(finding) for finding in comparison.missing_from_new],
                "unexpected_in_new": [asdict(finding) for finding in comparison.unexpected_in_new],
                "mismatched": [],
            }
            for comparison in result.comparisons
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity harness CLI."""
    parser = argparse.ArgumentParser(description="Compare legacy CICD enforcer output with elspeth-lints output")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    result = run_parity(load_manifest(args.manifest), root=args.root)
    if args.format == "json":
        sys.stdout.write(render_json(result))
    else:
        sys.stdout.write(render_text(result))
    return 0 if result.ok else 1


def _old_command_template(rule: MigrationRule) -> tuple[str, ...]:
    if rule.old_command is not None:
        return rule.old_command
    return (sys.executable, "{old_script}", "check", "--root", "{root}", "--format", "json")


def _new_command_template(rule: MigrationRule) -> tuple[str, ...]:
    if rule.new_command is not None:
        return rule.new_command
    return (
        sys.executable,
        "-m",
        "elspeth_lints.core.cli",
        "check",
        "--rules",
        "{new_rule}",
        "--root",
        "{root}",
        "--format",
        "json",
    )


def _expanded_command(template: Iterable[str], *, rule: MigrationRule, root: Path) -> tuple[str, ...]:
    replacements = {
        "{python}": sys.executable,
        "{root}": str(root),
        "{old_script}": rule.old_script,
        "{new_rule}": rule.new_rule or "",
    }
    expanded: list[str] = []
    for token in template:
        value = token
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        expanded.append(value)
    return tuple(expanded)


def _run_command(argv: tuple[str, ...], *, root: Path) -> CommandResult:
    completed = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    if completed.returncode not in SUCCESS_CODES:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(argv)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return CommandResult(argv=argv, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _normalize_findings(raw_stdout: str, *, source: str) -> tuple[NormalizedFinding, ...]:
    text = raw_stdout.strip()
    if not text:
        return ()
    payload = json.loads(text)
    findings_raw = _finding_list(payload, source=source)
    findings = [
        NormalizedFinding(
            file_path=_string_from_keys(item, ("file_path", "path", "file"), context=f"{source}[{index}]"),
            line=_int_from_keys(item, ("line", "lineno"), context=f"{source}[{index}]"),
            column=_int_from_keys(item, ("column", "col", "col_offset"), context=f"{source}[{index}]"),
            rule_id=_string_from_keys(item, ("rule_id", "rule", "id"), context=f"{source}[{index}]"),
            fingerprint=_string_from_keys(item, ("fingerprint", "key"), context=f"{source}[{index}]"),
            message=_string_from_keys(item, ("message", "description"), context=f"{source}[{index}]"),
        )
        for index, item in enumerate(findings_raw)
    ]
    return tuple(sorted(findings))


def _finding_list(payload: object, *, source: str) -> list[dict[str, Any]]:
    raw_findings: object
    if isinstance(payload, list):
        raw_findings = payload
    elif isinstance(payload, dict) and "findings" in payload:
        raw_findings = payload["findings"]
    elif isinstance(payload, dict) and "violations" in payload:
        raw_findings = payload["violations"]
    else:
        raise ValueError(f"{source}: JSON must be a list of findings or a mapping with 'findings'/'violations'")
    if not isinstance(raw_findings, list):
        raise ValueError(f"{source}: findings must be a list")
    findings: list[dict[str, Any]] = []
    for index, raw_finding in enumerate(raw_findings):
        findings.append(_mapping_value(raw_finding, f"{source}[{index}]"))
    return findings


def _finding_text(finding: NormalizedFinding) -> str:
    return f"{finding.file_path}:{finding.line}:{finding.column}:{finding.rule_id}: {finding.message}"


def _mapping_value(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _list_value(data: dict[str, Any], key: str, *, context: str) -> list[object]:
    if key not in data:
        return []
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{context}.{key} must be a list")
    return value


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    if key not in data:
        raise ValueError(f"{context} must include {key!r}")
    value = data[key]
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{context}.{key} must be a non-empty string")


def _optional_string(data: dict[str, Any], key: str, *, context: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{context}.{key} must be a non-empty string, null, or absent")


def _required_int(data: dict[str, Any], key: str, *, context: str) -> int:
    if key not in data:
        raise ValueError(f"{context} must include {key!r}")
    value = data[key]
    if isinstance(value, int):
        return value
    raise ValueError(f"{context}.{key} must be an integer")


def _optional_string_tuple(data: dict[str, Any], key: str, *, context: str) -> tuple[str, ...] | None:
    if key not in data or data[key] is None:
        return None
    values = _list_value(data, key, context=context)
    strings: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context}.{key}[{index}] must be a non-empty string")
        strings.append(value)
    return tuple(strings)


def _string_from_keys(data: dict[str, Any], keys: tuple[str, ...], *, context: str) -> str:
    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, str):
                return value
            raise ValueError(f"{context}.{key} must be a string")
    expected = ", ".join(keys)
    raise ValueError(f"{context} must include one of: {expected}")


def _int_from_keys(data: dict[str, Any], keys: tuple[str, ...], *, context: str) -> int:
    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, int):
                return value
            raise ValueError(f"{context}.{key} must be an integer")
    expected = ", ".join(keys)
    raise ValueError(f"{context} must include one of: {expected}")


if __name__ == "__main__":
    raise SystemExit(main())
